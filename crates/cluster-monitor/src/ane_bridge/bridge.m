// ANE FFI Bridge — Thin wrapper around Apple's private ANE APIs.
//
// EXPERIMENTAL: Uses _ANEInMemoryModel and related private classes from
// AppleNeuralEngine.framework. These APIs are undocumented and can break
// on any macOS update.
//
// License: MIT (same as asmi)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/runtime.h>
#import <objc/message.h>

// ---------------------------------------------------------------------------
// Runtime class references (resolved lazily)
// ---------------------------------------------------------------------------

static Class cls_InMemoryModelDescriptor = nil;
static Class cls_InMemoryModel = nil;
static Class cls_IOSurfaceObject = nil;
static Class cls_Request = nil;
static Class cls_PerfStats = nil;

static bool ane_classes_loaded = false;

// ---------------------------------------------------------------------------
// Public C API (called from Rust via FFI)
// ---------------------------------------------------------------------------

/// Initialize: load the private framework and resolve classes.
/// Returns 1 on success, 0 if ANE framework is unavailable.
int ane_bridge_init(void) {
    if (ane_classes_loaded) return 1;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    );
    if (!handle) return 0;

    cls_InMemoryModelDescriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    cls_InMemoryModel = NSClassFromString(@"_ANEInMemoryModel");
    cls_IOSurfaceObject = NSClassFromString(@"_ANEIOSurfaceObject");
    cls_Request = NSClassFromString(@"_ANERequest");
    cls_PerfStats = NSClassFromString(@"_ANEPerformanceStats");

    if (!cls_InMemoryModel || !cls_InMemoryModelDescriptor) return 0;

    ane_classes_loaded = true;
    return 1;
}

/// Check if ANE bridge is available.
int ane_bridge_available(void) {
    return ane_classes_loaded ? 1 : 0;
}

/// Opaque handle to a compiled ANE kernel.
typedef struct {
    id model;           // _ANEInMemoryModel
    id request;         // _ANERequest (cached)
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int n_inputs;
    int n_outputs;
} ANEKernelHandle;

/// Create an IOSurface of the given byte size.
static IOSurfaceRef create_surface(size_t bytes) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfacePixelFormat: @0x00000000,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

/// Compile a MIL program with baked weights into an ANE kernel.
/// Returns an opaque handle, or NULL on failure.
///
/// - mil_text: UTF-8 MIL program text
/// - mil_len: byte length of mil_text
/// - weight_data: binary weight blob (fp16 with headers)
/// - weight_len: byte length of weight_data
/// - input_sizes: array of byte sizes for each input IOSurface
/// - n_inputs: number of inputs
/// - output_sizes: array of byte sizes for each output IOSurface
/// - n_outputs: number of outputs
void *ane_bridge_compile(
    const char *mil_text, size_t mil_len,
    const uint8_t *weight_data, size_t weight_len,
    const size_t *input_sizes, int n_inputs,
    const size_t *output_sizes, int n_outputs
) {
    if (!ane_classes_loaded) return NULL;

    @autoreleasepool {
        NSString *mil = [[NSString alloc] initWithBytes:mil_text
                                                 length:mil_len
                                               encoding:NSUTF8StringEncoding];
        NSData *weights = [NSData dataWithBytes:weight_data length:weight_len];

        // Create model descriptor
        id descriptor = ((id (*)(id, SEL, id, id, id))objc_msgSend)(
            (id)cls_InMemoryModelDescriptor,
            NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
            mil, weights, nil
        );
        if (!descriptor) return NULL;

        // Create in-memory model
        id model = ((id (*)(id, SEL, id))objc_msgSend)(
            (id)cls_InMemoryModel,
            NSSelectorFromString(@"inMemoryModelWithDescriptor:"),
            descriptor
        );
        if (!model) return NULL;

        // Compile
        NSError *error = nil;
        BOOL ok = ((BOOL (*)(id, SEL, int, id, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"compileWithQoS:options:error:"),
            0, nil, &error
        );
        if (!ok) return NULL;

        // Load
        ok = ((BOOL (*)(id, SEL, int, id, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"loadWithQoS:options:error:"),
            0, nil, &error
        );
        if (!ok) return NULL;

        // Allocate handle
        ANEKernelHandle *h = calloc(1, sizeof(ANEKernelHandle));
        h->model = (__bridge_retained id)model;
        h->n_inputs = n_inputs;
        h->n_outputs = n_outputs;

        // Create IOSurfaces
        h->inputs = calloc(n_inputs, sizeof(IOSurfaceRef));
        h->outputs = calloc(n_outputs, sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++) {
            h->inputs[i] = create_surface(input_sizes[i]);
        }
        for (int i = 0; i < n_outputs; i++) {
            h->outputs[i] = create_surface(output_sizes[i]);
        }

        // Cache the evaluation request
        NSMutableArray *inputObjs = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *outputObjs = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:n_outputs];

        for (int i = 0; i < n_inputs; i++) {
            id obj = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                (id)cls_IOSurfaceObject,
                NSSelectorFromString(@"objectWithIOSurface:"),
                h->inputs[i]
            );
            [inputObjs addObject:obj];
            [inputIndices addObject:@(i)];
        }
        for (int i = 0; i < n_outputs; i++) {
            id obj = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                (id)cls_IOSurfaceObject,
                NSSelectorFromString(@"objectWithIOSurface:"),
                h->outputs[i]
            );
            [outputObjs addObject:obj];
            [outputIndices addObject:@(i)];
        }
        id request = ((id (*)(id, SEL, id, id, id, id, id, id, int))objc_msgSend)(
            (id)cls_Request,
            NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            inputObjs, inputIndices, outputObjs, outputIndices, nil, nil, 0
        );
        h->request = (__bridge_retained id)request;

        return h;
    }
}

/// Write data to an input IOSurface.
void ane_bridge_write_input(void *handle, int idx, const void *data, size_t bytes) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h || idx < 0 || idx >= h->n_inputs) return;

    IOSurfaceLock(h->inputs[idx], 0, NULL);
    void *base = IOSurfaceGetBaseAddress(h->inputs[idx]);
    memcpy(base, data, bytes);
    IOSurfaceUnlock(h->inputs[idx], 0, NULL);
}

/// Evaluate the compiled ANE kernel. Returns 1 on success, 0 on failure.
int ane_bridge_eval(void *handle) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h || !h->request) return 0;

    @autoreleasepool {
        // Evaluate directly using the cached request
        NSError *error = nil;
        BOOL ok = ((BOOL (*)(id, SEL, int, id, id, NSError **))objc_msgSend)(
            (__bridge id)h->model,
            NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
            0, nil, (__bridge id)h->request, &error
        );
        return ok ? 1 : 0;
    }
}

/// Read data from an output IOSurface.
void ane_bridge_read_output(void *handle, int idx, void *data, size_t bytes) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h || idx < 0 || idx >= h->n_outputs) return;

    IOSurfaceLock(h->outputs[idx], kIOSurfaceLockReadOnly, NULL);
    void *base = IOSurfaceGetBaseAddress(h->outputs[idx]);
    memcpy(data, base, bytes);
    IOSurfaceUnlock(h->outputs[idx], kIOSurfaceLockReadOnly, NULL);
}

/// Free a compiled ANE kernel and all its IOSurfaces.
void ane_bridge_free(void *handle) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h) return;

    @autoreleasepool {
        // Unload model
        id model = (__bridge_transfer id)h->model;
        ((void (*)(id, SEL, int, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"unloadWithQoS:error:"),
            0, nil
        );

        if (h->request) {
            id request = (__bridge_transfer id)h->request;
            (void)request; // ARC releases it
        }

        for (int i = 0; i < h->n_inputs; i++) {
            if (h->inputs[i]) CFRelease(h->inputs[i]);
        }
        for (int i = 0; i < h->n_outputs; i++) {
            if (h->outputs[i]) CFRelease(h->outputs[i]);
        }
        free(h->inputs);
        free(h->outputs);
        free(h);
    }
}
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSData, NSDictionary, NSNumber, NSQualityOfService, NSString};
use objc2_io_surface::IOSurface;

use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;
use crate::executable::Executable;
use crate::graph::Graph;
use crate::io_surface::IOSurfaceExt;
use crate::Error;

static FRAMEWORK_INIT: Once = Once::new();
static FRAMEWORK_OK: AtomicBool = AtomicBool::new(false);

fn ensure_framework() -> Result<(), Error> {
    FRAMEWORK_INIT.call_once(|| {
        let Ok(path) = CString::new(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        ) else {
            return;
        };
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW) };
        if !handle.is_null() {
            FRAMEWORK_OK.store(true, Ordering::Release);
        }
    });
    if FRAMEWORK_OK.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(Error::FrameworkLoad)
    }
}

fn nsdata_on_surface(data: &[u8]) -> (Retained<NSData>, Retained<IOSurface>) {
    let surface = IOSurface::with_byte_count(data.len());
    surface.write_bytes(data);
    let nsdata = unsafe {
        NSData::dataWithBytesNoCopy_length_freeWhenDone(
            surface.baseAddress(),
            data.len(),
            false,
        )
    };
    (nsdata, surface)
}

pub(crate) fn compile_network(
    graph: &Graph,
    quality_of_service: NSQualityOfService,
) -> Result<Executable, Error> {
    ensure_framework()?;

    let (ops, shapes) = graph.to_ops_and_shapes();
    let (mil_text, weight_bytes) = crate::ops::mil::emit_mil(&ops, &shapes);

    let (mil_data, _mil_surface) = nsdata_on_surface(mil_text.as_bytes());

    let _weight_surface: Option<Retained<IOSurface>>;
    let weights_dict: Retained<NSDictionary<NSString, AnyObject>> = if weight_bytes.is_empty() {
        _weight_surface = None;
        NSDictionary::new()
    } else {
        let (weight_data, weight_surface) = nsdata_on_surface(&weight_bytes);
        _weight_surface = Some(weight_surface);
        let offset = NSNumber::new_u64(0);
        let entry: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
            &[
                &*NSString::from_str("offset"),
                &*NSString::from_str("data"),
            ],
            &[
                offset.as_ref() as &AnyObject,
                weight_data.as_ref() as &AnyObject,
            ],
        );
        let key = NSString::from_str("@model_path/weights/weight.bin");
        NSDictionary::from_slices(&[&*key], &[entry.as_ref() as &AnyObject])
    };

    let descriptor = ANEInMemoryModelDescriptor::new(&mil_data, Some(&weights_dict))
        .ok_or(Error::ModelCreation)?;

    let model = ANEInMemoryModel::with_descriptor(&descriptor).ok_or(Error::ModelCreation)?;

    if let Some(hex_id) = model.hex_string_identifier() {
        let model_dir = std::env::temp_dir().join(hex_id.to_string());
        std::fs::create_dir_all(&model_dir)?;
        std::fs::write(model_dir.join("model.mil"), mil_text.as_bytes())?;
        if !weight_bytes.is_empty() {
            let weights_dir = model_dir.join("weights");
            std::fs::create_dir_all(&weights_dir)?;
            std::fs::write(weights_dir.join("weight.bin"), &weight_bytes)?;
        }
    }

    model
        .compile(quality_of_service)
        .map_err(|error| Error::Compile(error.localizedDescription().to_string()))?;

    model
        .load(quality_of_service)
        .map_err(|error| Error::Load(error.localizedDescription().to_string()))?;

    Ok(Executable {
        inner: model,
        qos: quality_of_service,
    })
}

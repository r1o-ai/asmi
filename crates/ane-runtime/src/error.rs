#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `AppleNeuralEngine.framework` could not be loaded via `dlopen`.
    #[error("failed to load AppleNeuralEngine.framework")]
    FrameworkLoad,

    /// The ANE daemon rejected the MIL program during compilation.
    #[error("ANE compile failed: {0}")]
    Compile(String),

    /// The ANE daemon failed to load the compiled program onto the hardware.
    #[error("ANE load failed: {0}")]
    Load(String),

    /// The ANE returned an error while evaluating a request.
    #[error("ANE evaluate failed: {0}")]
    Evaluate(String),

    /// `_ANERequest` could not be created from the provided IOSurfaces.
    #[error("failed to create ANERequest")]
    RequestCreation,

    /// An IOSurface could not be wrapped in an `_ANEIOSurfaceObject`.
    #[error("failed to wrap IOSurface for ANE")]
    SurfaceWrap,

    /// `_ANEInMemoryModel` could not be instantiated from the descriptor.
    #[error("failed to create _ANEInMemoryModel")]
    ModelCreation,

    /// A placeholder tensor has a spatial width below the ANE hardware minimum.
    #[error(
        "placeholder \"{name}\" has spatial width {width}, \
         but ANE requires at least {min} (pad the width dimension to {min} or larger)"
    )]
    SpatialWidthTooSmall {
        name: String,
        width: usize,
        min: usize,
    },

    /// A filesystem operation failed (writing MIL files to the temp directory).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

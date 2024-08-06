# video-capture

[![Version](https://img.shields.io/crates/v/video-capture)](https://crates.io/crates/video-capture)
[![Documentation](https://docs.rs/video-capture/badge.svg)](https://docs.rs/video-capture)
[![License](https://img.shields.io/badge/License-Apache%202-blue.svg)](LICENSE-APACHE)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE-MIT)

This project provides a cross-platform abstraction for capturing multimedia content from camera.

## Getting Started

To get started, add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
video-capture = "0.1"
```

```rust
// Create a default instance of camera manager
let mut cam_mgr = match CameraManager::default() {
    Ok(cam_mgr) => cam_mgr,
    Err(e) => {
        println!("{:?}", e.to_string());
        return;
    }
};

// List all camera devices
let devices = cam_mgr.list();
for device in devices {
    println!("name: {}", device.name());
    println!("id: {}", device.id());
}

// Get the first camera
let device = match cam_mgr.index_mut(0) {
    Some(device) => device,
    None => {
        println!("no camera found");
        return;
    }
};

// Set the output handler for the camera
if let Err(e) = device.set_output_handler(|frame| {
    println!("frame source: {:?}", frame.source);
    println!("frame desc: {:?}", frame.description());
    println!("frame timestamp: {:?}", frame.timestamp);

    Ok(())
}) {
    println!("{:?}", e.to_string());
};

// Map the video frame for memory access
if let Ok(mapped_guard) = frame.map() {
    if let Some(planes) = mapped_guard.planes() {
        for plane in planes {
            let plane_stride = plane.stride();
            let plane_height = plane.height();
            let plane_data = plane.data();
        }
    }
}

// Create a video frame that can be sent across threads
let shared_frame = SharedMediaFrame::new(frame.into_owned());

// Get supported formats
let formats = device.formats();
if let Ok(formats) = formats {
    if let Some(iter) = formats.array_iter() {
        for format in iter {
            println!("format: {:?}", format["format"]);
            println!("color-range: {:?}", format["color-range"]);
            println!("width: {:?}", format["width"]);
            println!("height: {:?}", format["height"]);
            println!("frame-rates: {:?}", format["frame-rates"]);
        }
    }
}

// Configure the camera
let mut option = Variant::new_dict();
option["width"] = 1280.into();
option["height"] = 720.into();
option["frame-rate"] = 30.0.into();
if let Err(e) = device.configure(option) {
    println!("{:?}", e.to_string());
}

// Start the camera
if let Err(e) = device.start() {
    println!("{:?}", e.to_string());
}

// Stop the camera
if let Err(e) = device.stop() {
    println!("{:?}", e.to_string());
}
```

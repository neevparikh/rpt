# Volumetric Light Transport Using Photon Mapping

You will need `rustc` and `cargo` to run. You can get them at https://rustup.rs/

## Results

<img width="416" alt="image" src="https://github.com/neevparikh/rpt/assets/41182432/d1d45e1b-1b1b-453a-a645-1c1dcb5d9fb8">
<img width="380" alt="image" src="https://github.com/neevparikh/rpt/assets/41182432/ad45e6f9-dd23-4135-aa54-e20ae6938f23">
<img width="367" alt="image" src="https://github.com/neevparikh/rpt/assets/41182432/4f80c44a-80d2-487d-87a7-80c1d03eba12">


## Running examples

```
cargo run --release --example volumetric_pathtrace_lampshade
cargo run --release --example volumetric_photonphoton_lampshade
cargo run --release --example volumetric_beamphoton_lampshade
cargo run --release --example volumetric_beambeam_lampshade
```

You can find outputs in the `lampshade` directory. You can edit parameters in the example files see (`examples/` directory). Note that only the examples above are made by us we, the rest are from base RPT and do not use volumetric materials.

## What we implemented

Four different integrators to render scenes with volumetric media. They are as follows:

- baseline volumetric path tracing
- baseline volumetric photon mapping (point queries with point photons)
- volumetric photon mapping with query beams (beam queries with point photons)
- volumetric photon mapping with query beams against beam photons (beam queries with beam photons)

To clarify: the variation comes in how you lookup into the photon map. With baseline volumetric photon mapping you pick a point in the volume and blur to find other points near it. With query beams, you find all photon points within a radius of the camera ray you send into the scene. With query beams _and_ photon beams, you store photons as line segments with a radius instead of single points, and you find all photon beams that pass near a camera ray you send into the scene. The advantage to this approach is you need far fewer photons in the map since the beams have a long extent.

## Where is the code

Our work is primarily in `src/photon.rs`, `src/medium.rs`, `src/renderer.rs`, `src/material.rs`.



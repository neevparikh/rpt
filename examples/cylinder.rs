use rpt::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut scene = Scene::new();

    scene.add(Object::new(
        load_stl("examples/cylinder.stl")?
            .translate(&glm::vec3(-15.0, -15.0, -25.0))
            .scale(&glm::vec3(1.0 / 15.0, 1.0 / 15.0, 1.0 / 25.0))
            .rotate_y(glm::quarter_pi()),
    ));
    scene.add(
        Object::new(plane(glm::vec3(0.0, 1.0, 0.0), -1.0))
            .material(Material::diffuse(hex_color(0xaaaaaa))),
    );

    scene.add(Light::Ambient(glm::vec3(0.1, 0.1, 0.1)));
    scene.add(Light::Point(
        glm::vec3(24.0, 24.0, 24.0),
        glm::vec3(0.0, 5.0, 5.0),
    ));
    scene.add(Light::Directional(
        glm::vec3(0.6, 0.6, 0.6),
        glm::vec3(1.0, -1.0, 0.0).normalize(),
    ));

    // black background
    scene.background = hex_color(0x000000);

    Renderer::new(&scene, Camera::default())
        .width(512)
        .height(512)
        .render()
        .save("output.png")?;

    Ok(())
}

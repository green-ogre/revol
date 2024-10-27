use crate::*;
use force_graph::{EdgeData, SimulationParameters};

pub async fn render_force_graph_edges(edges: &[NeuralCon]) {
    const NODE_RADIUS: f32 = 30.0;
    use force_graph::{ForceGraph, Node, NodeData};

    let mut graph = ForceGraph::<Neuron, f32>::new(SimulationParameters {
        force_charge: 12000.0,
        force_spring: 0.3,
        force_max: 280.0,
        node_speed: 7000.0,
        damping_factor: 0.4,
    });

    let mut added_neurons = Vec::with_capacity(edges.len());
    let angle_step = 2.0 * std::f32::consts::PI / (edges.len() as f32 + 1.);
    let radius = screen_width() / 6.;
    let mut i = 0;
    for edge in edges.iter() {
        let lhs_is_anchor = edges
            .iter()
            .filter(|e| e.lhs == edge.lhs || e.rhs == edge.lhs)
            .count()
            == 1;
        let rhs_is_anchor = edges
            .iter()
            .filter(|e| e.rhs == edge.rhs || e.lhs == edge.rhs)
            .count()
            == 1;

        let one = if let Some(idx) =
            added_neurons
                .iter()
                .find_map(|(n, idx)| if *n == edge.lhs { Some(idx) } else { None })
        {
            *idx
        } else {
            let angle = angle_step * i as f32;
            let x = radius * angle.cos() + screen_width() / 2.;
            let y = radius * angle.sin() + screen_height() / 2.;

            let node = NodeData {
                x,
                y,
                mass: 10.0,
                is_anchor: lhs_is_anchor,
                user_data: edge.lhs,
            };

            let idx = graph.add_node(node);
            added_neurons.push((edge.lhs, idx));
            idx
        };

        let two = if let Some(idx) =
            added_neurons
                .iter()
                .find_map(|(n, idx)| if *n == edge.rhs { Some(idx) } else { None })
        {
            *idx
        } else {
            let angle = angle_step * (i as f32 + 1.);
            let x = radius * angle.cos() + screen_width() / 2.;
            let y = radius * angle.sin() + screen_height() / 2.;

            let node = NodeData {
                x,
                y,
                mass: 10.0,
                is_anchor: rhs_is_anchor,
                user_data: edge.rhs,
            };

            let idx = graph.add_node(node);
            added_neurons.push((edge.rhs, idx));
            idx
        };

        graph.add_edge(
            one,
            two,
            EdgeData {
                user_data: edge.assoc.weight(),
            },
        );

        i += 2;
    }

    let mut dragging_node_idx = None;
    loop {
        if is_key_pressed(KeyCode::Escape) {
            exit(0);
        }

        if is_key_pressed(KeyCode::S) {
            break;
        }

        clear_background(Color::new(0.15, 0.15, 0.15, 1.));

        // draw edges
        graph.visit_edges(|node1, node2, edge| {
            let red = if edge.user_data > 0. { 1. } else { 0. };
            let green = if edge.user_data <= 0. { 1. } else { 0. };
            let thickness = if edge.user_data > 0. {
                edge.user_data / 4.
            } else {
                edge.user_data / -4.
            };

            draw_line(
                node1.x(),
                node1.y(),
                node2.x(),
                node2.y(),
                4. * (thickness + 1.),
                Color::new(red, green, 0., 1.),
            );
        });

        // draw nodes
        graph.visit_nodes(|node| {
            let color = match node.data.user_data {
                Neuron::Action(_) => MAROON,
                Neuron::Internal(_) => GRAY,
                Neuron::Sense(_) => DARKGREEN,
            };

            draw_circle(node.x(), node.y(), NODE_RADIUS, color);
            let name = format!("{:?}", node.data.user_data);
            draw_text(
                &name,
                node.x() - name.len() as f32 * 4.,
                node.y(),
                20.,
                WHITE,
            );

            // highlight hovered or dragged node
            if node_overlaps_mouse_position(node)
                || dragging_node_idx
                    .filter(|idx| *idx == node.index())
                    .is_some()
            {
                draw_circle_lines(node.x(), node.y(), NODE_RADIUS, 2.0, RED);
            }
        });

        // drag nodes with the mouse
        if is_mouse_button_down(MouseButton::Left) {
            graph.visit_nodes_mut(|node| {
                if let Some(idx) = dragging_node_idx {
                    if idx == node.index() {
                        let (mouse_x, mouse_y) = mouse_position();
                        node.data.x = mouse_x;
                        node.data.y = mouse_y;
                    }
                } else if node_overlaps_mouse_position(node) {
                    dragging_node_idx = Some(node.index());
                }
            });
        } else {
            dragging_node_idx = None;
        }

        graph.update(get_frame_time());

        next_frame().await
    }

    fn node_overlaps_mouse_position(node: &Node<Neuron>) -> bool {
        let (mouse_x, mouse_y) = mouse_position();
        ((node.x() - mouse_x) * (node.x() - mouse_x) + (node.y() - mouse_y) * (node.y() - mouse_y))
            < NODE_RADIUS * NODE_RADIUS
    }
}

pub async fn render_many_force_graph_edges(
    organisms: &[Organism],
    brains: &[&Brain],
    world: &World,
) {
    const NODE_RADIUS: f32 = 30.0;
    use force_graph::{ForceGraph, Node, NodeData};

    let build_graph = |edges: &[NeuralCon]| {
        let mut graph = ForceGraph::<Neuron, f32>::new(SimulationParameters {
            force_charge: 12000.0,
            force_spring: 0.3,
            force_max: 280.0,
            node_speed: 7000.0,
            damping_factor: 0.4,
        });

        let mut added_neurons = Vec::with_capacity(edges.len());
        let angle_step = 2.0 * std::f32::consts::PI / (edges.len() as f32 + 1.);
        let radius = screen_width() / 6.;
        let mut i = 0;

        for edge in edges.iter() {
            let lhs_is_anchor = edges
                .iter()
                .filter(|e| e.lhs == edge.lhs || e.rhs == edge.lhs)
                .count()
                == 1;
            let rhs_is_anchor = edges
                .iter()
                .filter(|e| e.rhs == edge.rhs || e.lhs == edge.rhs)
                .count()
                == 1;

            let one = if let Some(idx) =
                added_neurons
                    .iter()
                    .find_map(|(n, idx)| if *n == edge.lhs { Some(idx) } else { None })
            {
                *idx
            } else {
                let angle = angle_step * i as f32;
                let x = radius * angle.cos() + screen_width() / 2.;
                let y = radius * angle.sin() + screen_height() / 2.;

                let node = NodeData {
                    x,
                    y,
                    mass: 10.0,
                    is_anchor: lhs_is_anchor,
                    user_data: edge.lhs,
                };

                let idx = graph.add_node(node);
                added_neurons.push((edge.lhs, idx));
                idx
            };

            let two = if let Some(idx) =
                added_neurons
                    .iter()
                    .find_map(|(n, idx)| if *n == edge.rhs { Some(idx) } else { None })
            {
                *idx
            } else {
                let angle = angle_step * (i as f32 + 1.);
                let x = radius * angle.cos() + screen_width() / 2.;
                let y = radius * angle.sin() + screen_height() / 2.;

                let node = NodeData {
                    x,
                    y,
                    mass: 10.0,
                    is_anchor: rhs_is_anchor,
                    user_data: edge.rhs,
                };

                let idx = graph.add_node(node);
                added_neurons.push((edge.rhs, idx));
                idx
            };

            graph.add_edge(
                one,
                two,
                EdgeData {
                    user_data: edge.assoc.weight(),
                },
            );

            i += 2;
        }

        graph
    };

    let mut index = 0;
    let mut graph = build_graph(&brains[index].edges);

    let mut dragging_node_idx = None;
    loop {
        if is_key_pressed(KeyCode::Escape) {
            exit(0);
        }

        if is_key_pressed(KeyCode::S) {
            break;
        }

        if is_key_pressed(KeyCode::Right) {
            index += 1;
            index = index.clamp(0, brains.len() - 1);
            graph = build_graph(&brains[index].edges);
        }

        if is_key_pressed(KeyCode::Left) {
            index = index.saturating_sub(1);
            graph = build_graph(&brains[index].edges);
        }

        clear_background(Color::new(0.15, 0.15, 0.15, 1.));

        // draw edges
        graph.visit_edges(|node1, node2, edge| {
            let red = if edge.user_data > 0. { 1. } else { 0. };
            let green = if edge.user_data <= 0. { 1. } else { 0. };
            let thickness = if edge.user_data > 0. {
                edge.user_data / 4.
            } else {
                edge.user_data / -4.
            };

            draw_line(
                node1.x(),
                node1.y(),
                node2.x(),
                node2.y(),
                4. * (thickness + 1.),
                Color::new(red, green, 0., 1.),
            );
        });

        // draw nodes
        graph.visit_nodes(|node| {
            let color = match node.data.user_data {
                Neuron::Action(_) => MAROON,
                Neuron::Internal(_) => GRAY,
                Neuron::Sense(_) => DARKGREEN,
            };

            draw_circle(node.x(), node.y(), NODE_RADIUS, color);
            let name = format!("{:?}", node.data.user_data);
            draw_text(
                &name,
                node.x() - name.len() as f32 * 4.,
                node.y(),
                20.,
                WHITE,
            );

            if let Neuron::Action(_) = node.data.user_data {
                let val = brains[index]
                    .outputs
                    .iter()
                    .find(|out| out.neuron == node.data.user_data)
                    .unwrap()
                    .fire(&organisms[index], &world.state);
                let val = format!("{val:.4}");
                draw_text(
                    &val,
                    node.x() - val.len() as f32 * 4.,
                    node.y() + 20.,
                    20.,
                    WHITE,
                );
            }

            // highlight hovered or dragged node
            if node_overlaps_mouse_position(node)
                || dragging_node_idx
                    .filter(|idx| *idx == node.index())
                    .is_some()
            {
                draw_circle_lines(node.x(), node.y(), NODE_RADIUS, 2.0, RED);
            }
        });

        // drag nodes with the mouse
        if is_mouse_button_down(MouseButton::Left) {
            graph.visit_nodes_mut(|node| {
                if let Some(idx) = dragging_node_idx {
                    if idx == node.index() {
                        let (mouse_x, mouse_y) = mouse_position();
                        node.data.x = mouse_x;
                        node.data.y = mouse_y;
                    }
                } else if node_overlaps_mouse_position(node) {
                    dragging_node_idx = Some(node.index());
                }
            });
        } else {
            dragging_node_idx = None;
        }

        graph.update(get_frame_time());

        draw_text(
            &format!("Brains: {}/{}", index, brains.len()),
            20.,
            20.,
            30.,
            WHITE,
        );

        next_frame().await
    }

    fn node_overlaps_mouse_position(node: &Node<Neuron>) -> bool {
        let (mouse_x, mouse_y) = mouse_position();
        ((node.x() - mouse_x) * (node.x() - mouse_x) + (node.y() - mouse_y) * (node.y() - mouse_y))
            < NODE_RADIUS * NODE_RADIUS
    }
}

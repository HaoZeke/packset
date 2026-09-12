//! Memory islands: the natural clusters of the link graph, and the set a cue
//! activates. A persona is a view over everything; an island is what one
//! task touches.

use std::collections::HashMap;

use crate::search::Record;

/// The link graph over one live set, as positions.
pub struct Graph {
    ids: Vec<String>,
    adjacency: Vec<Vec<usize>>,
}

impl Graph {
    /// Symmetric edges from every atom's `links`; a link to an id outside the
    /// set is dropped.
    #[must_use]
    pub fn from_atoms(atoms: &[Record]) -> Self {
        let ids: Vec<String> = atoms
            .iter()
            .map(|a| {
                a.get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        let index: HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
        for (i, atom) in atoms.iter().enumerate() {
            let Some(links) = atom.get("links").and_then(|v| v.as_array()) else {
                continue;
            };
            for link in links.iter().filter_map(|l| l.as_str()) {
                if let Some(&j) = index.get(link) {
                    if i != j {
                        adjacency[i].push(j);
                        adjacency[j].push(i);
                    }
                }
            }
        }
        for row in &mut adjacency {
            row.sort_unstable();
            row.dedup();
        }
        Self { ids, adjacency }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[must_use]
    pub fn id(&self, at: usize) -> &str {
        &self.ids[at]
    }

    #[must_use]
    pub fn position(&self, id: &str) -> Option<usize> {
        self.ids.iter().position(|held| held == id)
    }
}

/// Rounds of label propagation before the labels are taken as they stand.
const PROPAGATION_ROUNDS: usize = 20;

/// The islands: communities by label propagation (Raghavan, Albert and
/// Kumara, doi:10.1103/PhysRevE.76.036106). Every node takes the label most
/// of its neighbours held in the previous round, smallest label on a tie, so
/// the answer is deterministic; a lone bridge edge loses to the clique on
/// its far side within two rounds. Largest island first, then by first
/// member.
#[must_use]
pub fn islands(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.len();
    let mut label: Vec<usize> = (0..n).collect();
    for _ in 0..PROPAGATION_ROUNDS {
        let next: Vec<usize> = (0..n)
            .map(|node| {
                let peers = &graph.adjacency[node];
                if peers.is_empty() {
                    return label[node];
                }
                let mut counts: HashMap<usize, usize> = HashMap::new();
                for &peer in peers {
                    *counts.entry(label[peer]).or_insert(0) += 1;
                }
                counts
                    .iter()
                    .map(|(&l, &c)| (c, std::cmp::Reverse(l)))
                    .max()
                    .map_or(label[node], |(_, std::cmp::Reverse(l))| l)
            })
            .collect();
        if next == label {
            break;
        }
        label = next;
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &l) in label.iter().enumerate() {
        groups.entry(l).or_default().push(node);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    for group in &mut out {
        group.sort_unstable();
    }
    out.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));
    out
}

/// How much of a node's activation reaches its neighbours per hop.
pub const HOP_DECAY: f64 = 0.5;

/// Spreading activation from weighted seeds (Collins and Loftus,
/// doi:10.1037/0033-295X.82.6.407; the fan effect of Anderson's 1983
/// spreading-activation theory of memory, as division by degree). Returns
/// every node that received activation, strongest first.
#[must_use]
pub fn activate(graph: &Graph, seeds: &[(usize, f64)], hops: usize) -> Vec<(usize, f64)> {
    let n = graph.len();
    let mut activation = vec![0.0f64; n];
    let mut frontier = vec![0.0f64; n];
    for &(node, weight) in seeds {
        if node < n && weight > 0.0 {
            activation[node] += weight;
            frontier[node] += weight;
        }
    }
    for _ in 0..hops {
        let mut next = vec![0.0f64; n];
        for (energy, peers) in frontier.iter().zip(&graph.adjacency) {
            if *energy <= 0.0 || peers.is_empty() {
                continue;
            }
            let share = HOP_DECAY * energy / peers.len() as f64;
            for &peer in peers {
                next[peer] += share;
            }
        }
        for (held, gained) in activation.iter_mut().zip(&next) {
            *held += gained;
        }
        frontier = next;
    }
    let mut out: Vec<(usize, f64)> = activation
        .into_iter()
        .enumerate()
        .filter(|(_, a)| *a > 0.0)
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn clique(prefix: &str, n: usize) -> Vec<Record> {
        (0..n)
            .map(|i| {
                let links: Vec<String> = (0..n)
                    .filter(|j| *j != i)
                    .map(|j| format!("{prefix}{j}"))
                    .collect();
                json!({"id": format!("{prefix}{i}"), "links": links})
                    .as_object()
                    .cloned()
                    .unwrap()
            })
            .collect()
    }

    /// Two cliques joined by one edge are two islands, and a cue in one
    /// activates its own clique above the other.
    #[test]
    fn two_cliques_are_two_islands() {
        let mut atoms = clique("a", 4);
        atoms.extend(clique("b", 4));
        atoms[0]["links"].as_array_mut().unwrap().push(json!("b0"));
        let graph = Graph::from_atoms(&atoms);
        let found = islands(&graph);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].len(), 4);
        let a1 = graph.position("a1").unwrap();
        let lit = activate(&graph, &[(a1, 1.0)], 2);
        let score = |id: &str| {
            lit.iter()
                .find(|(n, _)| graph.id(*n) == id)
                .map_or(0.0, |(_, a)| *a)
        };
        for own in ["a0", "a2", "a3"] {
            for other in ["b1", "b2", "b3"] {
                assert!(score(own) > score(other), "{own} {other} {lit:?}");
            }
        }
        assert_eq!(lit[0].0, a1);
    }

    /// An atom with no links is its own island and activates nothing else.
    #[test]
    fn a_lone_atom_is_an_island() {
        let atoms = vec![json!({"id": "solo"}).as_object().cloned().unwrap()];
        let graph = Graph::from_atoms(&atoms);
        assert_eq!(islands(&graph), vec![vec![0]]);
        assert_eq!(activate(&graph, &[(0, 1.0)], 3), vec![(0, 1.0)]);
    }
}

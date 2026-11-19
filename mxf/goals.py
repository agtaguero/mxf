"""LIBERO goal predicates over MotrixSim positions, joints and collision pairs."""

import itertools
import operator

import numpy as np


class TaskGoals:
    def __init__(self, model, config):
        objects = config["objects"]
        self.links = {name: model.get_link(spec["body"]) for name, spec in objects.items()}
        self.sites = {
            name: (model.get_site(name), np.array(spec["size"]), spec["parent"])
            for name, spec in config.get("sites", {}).items()
        }
        self.rules = {}
        for key, rule in config.get("joint_rules", {}).items():
            self.rules[key] = (
                [model.get_joint(j) for j in rule["joints"]],
                getattr(operator, rule["op"]),
                rule["limit"],
            )
        self.predicates = config["goals"]
        self.contacts = {}
        for predicate, *terms in self.predicates:
            if predicate == "on":
                subject, target = terms
                parent = self.sites[target][2] if target in self.sites else target
                if parent in objects:
                    self.contacts[subject, target] = np.array(
                        list(
                            itertools.product(
                                *(
                                    tuple(model.get_geom_index(n) for n in objects[key]["contact_geoms"])
                                    for key in (subject, parent)
                                )
                            )
                        ),
                        dtype=np.uint32,
                    )
            elif predicate == "in" and terms[1] in self.sites:
                continue
            elif "/".join([predicate, *terms]) not in self.rules:
                raise ValueError(f"Unsupported LIBERO goal {predicate} {terms}")

    def satisfied(self, data, query):
        for predicate, *terms in self.predicates:
            if len(terms) == 1:
                joints, compare, threshold = self.rules[predicate + "/" + terms[0]]
                results = (compare(j.get_dof_pos(data).item(), threshold) for j in joints)
                if not (all(results) if predicate == "close" else any(results)):
                    return False
                continue
            subject, target = terms
            position = self.links[subject].get_position(data)
            if target in self.sites:
                site, size, _ = self.sites[target]
                center = site.get_position(data)
                # LIBERO SiteObject.under multiplies by the site rotation itself.
                rotation = site.get_rotation_mat(data)
                if predicate == "in":
                    span = np.abs(rotation @ size)
                    if not (
                        np.all(position > center - span - [0, 0, 0.01]) and np.all(position < center + span)
                    ):
                        return False
                else:
                    delta = rotation @ (position - center)
                    if not (
                        size[2] - 0.005 < delta[2] < size[2] + 0.10 and np.all(np.abs(delta[:2]) < size[:2])
                    ):
                        return False
            else:
                base = self.links[target].get_position(data)
                if not (base[2] <= position[2] and np.linalg.norm(position[:2] - base[:2]) < 0.03):
                    return False
            pairs = self.contacts.get((subject, target))
            if pairs is not None and not query.is_colliding(pairs).any():
                return False
        return True

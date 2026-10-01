"""Streaming reader for the supported metabolic subset of SBML."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from typing import BinaryIO

from ._base import _open_binary

__all__ = ["MetabolicModel", "read_sbml"]

CORE_V1 = "http://www.sbml.org/sbml/level3/version1/core"
FBC_V2 = "http://www.sbml.org/sbml/level3/version1/fbc/version2"
GROUPS_V1 = "http://www.sbml.org/sbml/level3/version1/groups/version1"
_SBML_PACKAGE_PREFIX = "http://www.sbml.org/sbml/level3/"
_SUPPORTED_NAMESPACES = {CORE_V1, FBC_V2, GROUPS_V1}
_RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"


@lru_cache(maxsize=256)
def _split_tag(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        namespace, local = tag[1:].split("}", 1)
        return namespace, local
    return "", tag


def _tag(element: ET.Element) -> tuple[str, str]:
    return _split_tag(element.tag)


def _bool(value: str | None, *, default: bool, where: str) -> bool:
    if value is None:
        return default
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean {value!r} on {where}")


def _number(value: str, *, where: str, allow_infinity: bool = True) -> float:
    special = {"INF": math.inf, "+INF": math.inf, "-INF": -math.inf}
    if value in special:
        if allow_infinity:
            return special[value]
        raise ValueError(f"Infinite value is not allowed for {where}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value {value!r} for {where}") from exc
    if math.isnan(parsed):
        raise ValueError(f"NaN is not allowed for {where}")
    if math.isinf(parsed) and not allow_infinity:
        raise ValueError(f"Infinite value is not allowed for {where}")
    return parsed


def _package_attr(element: ET.Element, namespace: str, name: str) -> str | None:
    return element.get(f"{{{namespace}}}{name}")


def _record_meta(element: ET.Element, annotations: bool, package: str | None = None) -> dict:
    metadata = {
        "id": _package_attr(element, package, "id") if package else element.get("id"),
        "name": _package_attr(element, package, "name") if package else element.get("name"),
        "meta_id": element.get("metaid"),
        "sbo_term": element.get("sboTerm"),
    }
    if annotations:
        metadata["annotations"] = _annotation_resources(element)
    return metadata


def _annotation_resources(element: ET.Element) -> tuple[str, ...]:
    resources: list[str] = []
    for node in element:
        if _tag(node) != (CORE_V1, "annotation"):
            continue
        for descendant in node.iter():
            value = descendant.get(_RDF_RESOURCE)
            if value is not None:
                resources.append(value)
    return tuple(resources)


@dataclass
class MetabolicModel:
    """Plain parsed SBML records, independent of CORNETO."""

    model_info: dict = field(default_factory=dict)
    compartments: dict[str, dict] = field(default_factory=dict)
    unit_definitions: dict[str, dict] = field(default_factory=dict)
    species: dict[str, dict] = field(default_factory=dict)
    parameters: dict[str, dict] = field(default_factory=dict)
    gene_products: dict[str, dict] = field(default_factory=dict)
    reactions: dict[str, dict] = field(default_factory=dict)
    objectives: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    active_objective: str | None = None


def _parse_gpr(element: ET.Element) -> tuple[str, set[str]]:
    namespace, kind = _tag(element)
    if namespace != FBC_V2:
        raise ValueError(f"Unsupported element in FBC gene association: {kind!r}")
    if kind == "geneProductAssociation":
        children = [child for child in element if _tag(child)[0] == FBC_V2]
        if len(children) != 1:
            raise ValueError("geneProductAssociation must contain exactly one rule")
        return _parse_gpr(children[0])
    if kind == "geneProductRef":
        gene_id = element.get(f"{{{FBC_V2}}}geneProduct")
        if not gene_id:
            raise ValueError("geneProductRef is missing its FBC geneProduct attribute")
        return gene_id, {gene_id}
    if kind in {"and", "or"}:
        children = [child for child in element if _tag(child)[0] == FBC_V2]
        if len(children) < 2:
            raise ValueError(f"FBC {kind} rules require at least two children")
        parsed = [_parse_gpr(child) for child in children]
        return (
            "(" + f" {kind} ".join(text for text, _ in parsed) + ")",
            set().union(*(genes for _, genes in parsed)),
        )
    raise ValueError(f"Unsupported FBC gene-reaction rule element: {kind!r}")


def _participants(reaction: ET.Element) -> tuple[dict, dict, list[dict]]:
    sides: dict[str, dict[str, float]] = {"reactants": {}, "products": {}}
    modifiers: list[dict] = []
    for side, list_name in (("reactants", "listOfReactants"), ("products", "listOfProducts")):
        reference_list = next(
            (child for child in reaction if _tag(child) == (CORE_V1, list_name)),
            None,
        )
        if reference_list is None:
            continue
        for ref in reference_list:
            if _tag(ref) != (CORE_V1, "speciesReference"):
                continue
            species_id = ref.get("species")
            if not species_id:
                raise ValueError(f"Reaction {reaction.get('id')!r} has a speciesReference without species")
            if ref.get("constant") is None:
                raise ValueError(f"Reaction {reaction.get('id')!r} speciesReference is missing required constant")
            if not _bool(ref.get("constant"), default=False, where="speciesReference.constant"):
                raise ValueError(f"Reaction {reaction.get('id')!r} has a non-constant speciesReference")
            if ref.get("stoichiometry") is None:
                raise ValueError(
                    f"Reaction {reaction.get('id')!r} has species {species_id!r} with unspecified stoichiometry"
                )
            coefficient = _number(
                ref.get("stoichiometry"), where=f"{reaction.get('id')}.stoichiometry", allow_infinity=False
            )
            if coefficient < 0:
                raise ValueError(f"Reaction {reaction.get('id')!r} has negative stoichiometry")
            if ref.find(f"{{{CORE_V1}}}stoichiometryMath") is not None:
                raise ValueError(f"Reaction {reaction.get('id')!r} uses stoichiometryMath, unsupported for FBA")
            sides[side][species_id] = sides[side].get(species_id, 0.0) + coefficient

    modifier_list = next(
        (child for child in reaction if _tag(child) == (CORE_V1, "listOfModifiers")),
        None,
    )
    if modifier_list is not None:
        for ref in modifier_list:
            if _tag(ref) != (CORE_V1, "modifierSpeciesReference"):
                continue
            species_id = ref.get("species")
            if not species_id:
                raise ValueError(f"Reaction {reaction.get('id')!r} has a modifier without species")
            modifiers.append(
                {
                    "species": species_id,
                    "id": ref.get("id"),
                    "meta_id": ref.get("metaid"),
                    "sbo_term": ref.get("sboTerm"),
                }
            )

    return sides["reactants"], sides["products"], modifiers


def _read_model(stream: BinaryIO, *, annotations: bool, source_name: str) -> MetabolicModel:
    model = MetabolicModel()
    # The public read_sbml context owns source cleanup; this parser only reads.
    stack: list[ET.Element] = []
    opaque_modes: list[str | None] = []
    skip_stack: list[bool] = []
    package_required: dict[str, bool] = {}
    root_element: ET.Element | None = None
    model_element: ET.Element | None = None
    root_seen = False
    fbc_seen = False
    groups_seen = False
    saw_objectives = False
    ids: dict[str, str] = {}
    meta_ids: dict[str, str] = {}

    def register(
        record: ET.Element,
        kind: str,
        identifier: str | None = None,
        *,
        global_id: bool = True,
    ) -> None:
        if identifier is None and global_id:
            identifier = record.get("id")
        if identifier:
            previous = ids.get(identifier)
            if previous is not None:
                raise ValueError(f"Duplicate SBML id {identifier!r} on {kind}; already used by {previous}")
            ids[identifier] = kind
        meta_id = record.get("metaid")
        if meta_id:
            previous = meta_ids.get(meta_id)
            if previous is not None:
                raise ValueError(f"Duplicate SBML metaid {meta_id!r} on {kind}; already used by {previous}")
            meta_ids[meta_id] = kind

    for event, element in ET.iterparse(stream, events=("start", "end")):
        if event == "start":
            parent = stack[-1] if stack else None
            parent_mode = opaque_modes[-1] if opaque_modes else None
            parent_skip = skip_stack[-1] if skip_stack else False

            # Opaque notes/annotations and unsupported optional packages
            # need no semantic tag handling below this point.
            if parent_mode is not None or parent_skip:
                stack.append(element)
                opaque_modes.append(parent_mode)
                skip_stack.append(parent_skip)
                continue

            namespace, kind = _tag(element)
            own_opaque = kind if namespace == CORE_V1 and kind in {"notes", "annotation"} else None
            opaque_mode = parent_mode or own_opaque
            skip = parent_skip

            if not root_seen:
                root_seen = True
                if kind != "sbml" or namespace != CORE_V1:
                    raise ValueError(f"Expected SBML Level 3 Version 1, found {{{namespace}}}{kind}")
                root_element = element
                if element.get("level") != "3" or element.get("version") != "1":
                    raise ValueError(
                        "Unsupported SBML level/version "
                        f"{element.get('level')!r}/{element.get('version')!r}; expected 3/1"
                    )
                for attr, value in element.attrib.items():
                    attr_ns, local = _tag(ET.Element(attr))
                    if local != "required":
                        continue
                    if "/fbc/" in attr_ns and attr_ns != FBC_V2:
                        raise ValueError(f"Unsupported SBML FBC package version: {attr_ns}")
                    if "/groups/" in attr_ns and attr_ns != GROUPS_V1:
                        raise ValueError(f"Unsupported SBML Groups package version: {attr_ns}")
                    required = _bool(value, default=False, where=f"{attr} package required")
                    if not attr_ns or (required and attr_ns not in {FBC_V2, GROUPS_V1}):
                        raise ValueError(f"Required SBML package is unsupported: {attr_ns or attr}")
                    package_required[attr_ns] = required
                    fbc_seen |= attr_ns == FBC_V2
                    groups_seen |= attr_ns == GROUPS_V1
            elif not parent_mode and not parent_skip and namespace.startswith(_SBML_PACKAGE_PREFIX):
                if "/fbc/" in namespace and namespace != FBC_V2:
                    raise ValueError(f"Unsupported SBML FBC package version: {namespace}")
                if "/groups/" in namespace and namespace != GROUPS_V1:
                    raise ValueError(f"Unsupported SBML Groups package version: {namespace}")
                if namespace not in _SUPPORTED_NAMESPACES:
                    required = package_required.get(namespace)
                    if required is None:
                        raise ValueError(f"SBML package has no root required declaration: {namespace}")
                    if required:
                        raise ValueError(f"Required SBML package is unsupported: {namespace}")
                    skip = True
            elif not parent_mode and not parent_skip and namespace not in _SUPPORTED_NAMESPACES:
                skip = True

            if (
                not opaque_mode
                and not skip
                and model_element is not None
                and model_element in stack
                and namespace == CORE_V1
                and kind
                in {
                    "listOfRules",
                    "listOfEvents",
                    "listOfInitialAssignments",
                    "listOfConstraints",
                    "initialAssignment",
                    "assignmentRule",
                    "rateRule",
                    "algebraicRule",
                    "event",
                    "constraint",
                    "stoichiometryMath",
                    "kineticLaw",
                }
            ):
                raise ValueError(f"SBML {kind} is unsupported in a static FBA model")

            if not skip and not opaque_mode and namespace == CORE_V1 and kind == "model":
                if parent is not root_element:
                    raise ValueError("SBML model must be a direct child of sbml")
                if model_element is not None:
                    raise ValueError("SBML document contains multiple core model elements")
                model_element = element
            if not skip and not opaque_mode:
                if namespace == FBC_V2:
                    fbc_seen = True
                elif namespace == GROUPS_V1:
                    groups_seen = True

            stack.append(element)
            opaque_modes.append(opaque_mode)
            skip_stack.append(skip)
            continue

        parent = stack[-2] if len(stack) > 1 else None
        opaque_mode = opaque_modes[-1]
        skip = skip_stack[-1]
        if skip:
            if parent is not None:
                parent.remove(element)
            element.clear()
            stack.pop()
            opaque_modes.pop()
            skip_stack.pop()
            continue
        if opaque_mode is not None:
            should_clear = opaque_mode == "notes" or (opaque_mode == "annotation" and not annotations)
            if should_clear:
                if parent is not None:
                    parent.remove(element)
                element.clear()
            stack.pop()
            opaque_modes.pop()
            skip_stack.pop()
            continue

        namespace, kind = _tag(element)
        parent_ns, parent_kind = _tag(parent) if parent is not None else ("", "")

        in_model = model_element is not None and any(item is model_element for item in stack[:-1])
        direct_list_child = len(stack) >= 3 and stack[-3] is model_element
        is_record = False

        if namespace == CORE_V1 and kind == "model" and parent is root_element:
            register(element, "model")
            model.model_info.update(_record_meta(element, annotations))
            for attr in ("substanceUnits", "timeUnits", "volumeUnits", "areaUnits", "lengthUnits", "extentUnits"):
                if element.get(attr) is not None:
                    model.model_info[attr] = element.get(attr)
            if element.get("conversionFactor"):
                raise ValueError("Model conversionFactor is unsupported in a static FBA model")
            strict_value = _package_attr(element, FBC_V2, "strict")
            if fbc_seen and strict_value is None:
                raise ValueError("FBC model is missing required fbc:strict")
            if strict_value is not None:
                model.model_info["fbc_strict"] = _bool(strict_value, default=False, where="model fbc:strict")
            is_record = True
        elif (
            in_model
            and namespace == CORE_V1
            and kind == "compartment"
            and parent_ns == CORE_V1
            and parent_kind == "listOfCompartments"
            and direct_list_child
        ):
            identifier = element.get("id")
            if not identifier:
                raise ValueError("SBML compartment is missing its id")
            if element.get("constant") is None:
                raise ValueError(f"Compartment {identifier!r} is missing required constant")
            register(element, "compartment")
            model.compartments[identifier] = {
                **_record_meta(element, annotations),
                "size": _number(element.get("size"), where=f"compartment {identifier}.size", allow_infinity=False)
                if element.get("size") is not None
                else None,
                "units": element.get("units"),
                "constant": _bool(element.get("constant"), default=False, where=f"compartment {identifier}.constant"),
            }
            is_record = True
        elif (
            in_model
            and namespace == CORE_V1
            and kind == "species"
            and parent_ns == CORE_V1
            and parent_kind == "listOfSpecies"
            and direct_list_child
        ):
            identifier = element.get("id")
            if not identifier:
                raise ValueError("SBML species is missing its id")
            compartment = element.get("compartment")
            if not compartment:
                raise ValueError(f"Species {identifier!r} is missing required compartment")
            for required_attr in ("boundaryCondition", "constant", "hasOnlySubstanceUnits"):
                if element.get(required_attr) is None:
                    raise ValueError(f"Species {identifier!r} is missing required {required_attr}")
            if element.get("initialAmount") is not None and element.get("initialConcentration") is not None:
                raise ValueError(f"Species {identifier!r} has both initialAmount and initialConcentration")
            charge_text = element.get(f"{{{FBC_V2}}}charge")
            charge = None
            if charge_text is not None:
                charge_value = _number(charge_text, where=f"species {identifier}.charge", allow_infinity=False)
                if not charge_value.is_integer():
                    raise ValueError(f"Species {identifier!r} has non-integer FBC charge {charge_text!r}")
                charge = int(charge_value)
            register(element, "species")
            model.species[identifier] = {
                **_record_meta(element, annotations),
                "compartment": compartment,
                "boundary_condition": _bool(
                    element.get("boundaryCondition"), default=False, where=f"species {identifier}.boundaryCondition"
                ),
                "constant": _bool(element.get("constant"), default=False, where=f"species {identifier}.constant"),
                "has_only_substance_units": _bool(
                    element.get("hasOnlySubstanceUnits"),
                    default=False,
                    where=f"species {identifier}.hasOnlySubstanceUnits",
                ),
                "initial_amount": _number(
                    element.get("initialAmount"), where=f"species {identifier}.initialAmount", allow_infinity=False
                )
                if element.get("initialAmount") is not None
                else None,
                "initial_concentration": _number(
                    element.get("initialConcentration"),
                    where=f"species {identifier}.initialConcentration",
                    allow_infinity=False,
                )
                if element.get("initialConcentration") is not None
                else None,
                "substance_units": element.get("substanceUnits"),
                "charge": charge,
                "formula": element.get(f"{{{FBC_V2}}}chemicalFormula"),
            }
            if element.get("conversionFactor"):
                raise ValueError(f"Species {identifier!r} conversionFactor is unsupported in static FBA")
            is_record = True
        elif (
            in_model
            and namespace == CORE_V1
            and kind == "parameter"
            and parent_ns == CORE_V1
            and parent_kind == "listOfParameters"
            and direct_list_child
        ):
            identifier = element.get("id")
            if not identifier:
                raise ValueError("SBML parameter is missing its id")
            if element.get("constant") is None:
                raise ValueError(f"Parameter {identifier!r} is missing required constant")
            register(element, "parameter")
            model.parameters[identifier] = {
                **_record_meta(element, annotations),
                "value": _number(element.get("value"), where=f"parameter {identifier}.value")
                if element.get("value") is not None
                else None,
                "constant": _bool(element.get("constant"), default=False, where=f"parameter {identifier}.constant"),
                "units": element.get("units"),
            }
            is_record = True
        elif (
            in_model
            and namespace == CORE_V1
            and kind == "unitDefinition"
            and parent_ns == CORE_V1
            and parent_kind == "listOfUnitDefinitions"
            and direct_list_child
        ):
            identifier = element.get("id")
            if not identifier:
                raise ValueError("SBML unitDefinition is missing its UnitSId")
            if identifier in model.unit_definitions:
                raise ValueError(f"Duplicate SBML UnitSId {identifier!r}")
            unit_list = element.find(f"{{{CORE_V1}}}listOfUnits")
            units = []
            for unit in unit_list if unit_list is not None else ():
                if _tag(unit) == (CORE_V1, "unit"):
                    units.append({key: unit.get(key) for key in ("kind", "exponent", "scale", "multiplier", "offset")})
            register(element, "unitDefinition", global_id=False)
            model.unit_definitions[identifier] = {**_record_meta(element, annotations), "units": units}
            is_record = True
        elif (
            in_model
            and namespace == FBC_V2
            and kind == "geneProduct"
            and parent_ns == FBC_V2
            and parent_kind == "listOfGeneProducts"
            and direct_list_child
        ):
            identifier = _package_attr(element, FBC_V2, "id")
            if not identifier:
                raise ValueError("FBC geneProduct is missing its id")
            register(element, "geneProduct", identifier)
            model.gene_products[identifier] = {
                **_record_meta(element, annotations, FBC_V2),
                "label": _package_attr(element, FBC_V2, "label"),
                "associated_species": _package_attr(element, FBC_V2, "associatedSpecies"),
            }
            is_record = True
        elif (
            in_model
            and namespace == CORE_V1
            and kind == "reaction"
            and parent_ns == CORE_V1
            and parent_kind == "listOfReactions"
            and direct_list_child
        ):
            identifier = element.get("id")
            if not identifier:
                raise ValueError("SBML reaction is missing its id")
            if element.get("reversible") is None:
                raise ValueError(f"Reaction {identifier!r} is missing required reversible")
            register(element, "reaction")
            reactants, products, modifiers = _participants(element)
            associations = [child for child in element if _tag(child) == (FBC_V2, "geneProductAssociation")]
            if len(associations) > 1:
                raise ValueError(f"Reaction {identifier!r} has more than one geneProductAssociation")
            gpr, gpr_genes = _parse_gpr(associations[0]) if associations else ("", set())
            lower_ref = _package_attr(element, FBC_V2, "lowerFluxBound")
            upper_ref = _package_attr(element, FBC_V2, "upperFluxBound")
            fast_text = element.get("fast")
            fast = (
                _bool(fast_text, default=False, where=f"reaction {identifier}.fast") if fast_text is not None else None
            )
            if fast is True:
                raise ValueError(f"Reaction {identifier!r} uses fast=true, unsupported in static FBA")
            model.reactions[identifier] = {
                **_record_meta(element, annotations),
                "reversible": _bool(
                    element.get("reversible"), default=False, where=f"reaction {identifier}.reversible"
                ),
                "fast": fast,
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "lower_bound_ref": lower_ref,
                "upper_bound_ref": upper_ref,
                "lower_bound": None,
                "upper_bound": None,
                "gpr": gpr,
                "gpr_genes": gpr_genes,
            }
            is_record = True
        elif in_model and namespace == FBC_V2 and kind == "listOfObjectives" and parent is model_element:
            if saw_objectives:
                raise ValueError("SBML model has more than one listOfObjectives")
            saw_objectives = True
            model.active_objective = _package_attr(element, FBC_V2, "activeObjective")
        elif (
            in_model
            and namespace == FBC_V2
            and kind == "objective"
            and parent_ns == FBC_V2
            and parent_kind == "listOfObjectives"
            and direct_list_child
        ):
            identifier = _package_attr(element, FBC_V2, "id")
            if not identifier:
                raise ValueError("FBC objective is missing its id")
            register(element, "objective", identifier)
            sense = _package_attr(element, FBC_V2, "type")
            if sense not in {"maximize", "minimize"}:
                raise ValueError(f"Objective {identifier!r} has invalid type {sense!r}")
            coefficients: dict[str, float] = {}
            flux_objectives = element.find(f"{{{FBC_V2}}}listOfFluxObjectives")
            for flux_objective in flux_objectives if flux_objectives is not None else ():
                if _tag(flux_objective) != (FBC_V2, "fluxObjective"):
                    continue
                reaction_id = _package_attr(flux_objective, FBC_V2, "reaction")
                coefficient_text = _package_attr(flux_objective, FBC_V2, "coefficient")
                if not reaction_id or coefficient_text is None:
                    raise ValueError(f"Objective {identifier!r} has incomplete fluxObjective")
                if reaction_id in coefficients:
                    raise ValueError(f"Objective {identifier!r} repeats reaction {reaction_id!r}")
                coefficients[reaction_id] = _number(
                    coefficient_text, where=f"objective {identifier}.{reaction_id}", allow_infinity=False
                )
            model.objectives[identifier] = {
                **_record_meta(element, annotations, FBC_V2),
                "sense": sense,
                "coefficients": coefficients,
            }
            is_record = True
        elif (
            in_model
            and namespace == GROUPS_V1
            and kind == "group"
            and parent_ns == GROUPS_V1
            and parent_kind == "listOfGroups"
            and direct_list_child
        ):
            identifier = _package_attr(element, GROUPS_V1, "id")
            if not identifier:
                raise ValueError("Groups group is missing its id")
            register(element, "group", identifier)
            members: list[dict[str, str]] = []
            member_list = element.find(f"{{{GROUPS_V1}}}listOfMembers")
            for member in member_list if member_list is not None else ():
                if _tag(member) == (GROUPS_V1, "member"):
                    id_ref = _package_attr(member, GROUPS_V1, "idRef")
                    meta_ref = _package_attr(member, GROUPS_V1, "metaIdRef")
                    if bool(id_ref) == bool(meta_ref):
                        raise ValueError(f"Group {identifier!r} member must have exactly one of idRef/metaIdRef")
                    members.append({"id_ref": id_ref, "meta_id_ref": meta_ref})
            model.groups[identifier] = {
                **_record_meta(element, annotations, GROUPS_V1),
                "kind": _package_attr(element, GROUPS_V1, "kind"),
                "members": members,
            }
            is_record = True

        # UnitSIds have their own namespace; all other SBML IDs are global.
        generic_id = element.get("id") if namespace == CORE_V1 else None
        if namespace == FBC_V2:
            generic_id = _package_attr(element, FBC_V2, "id")
        elif namespace == GROUPS_V1:
            generic_id = _package_attr(element, GROUPS_V1, "id")
        generic_meta = element.get("metaid")
        if in_model and not is_record:
            if kind == "unitDefinition":
                generic_id = None
            if generic_id or generic_meta:
                register(element, f"SBML {kind}", generic_id, global_id=generic_id is not None)

        if is_record:
            if parent is not None:
                parent.remove(element)
            element.clear()
        stack.pop()
        opaque_modes.pop()
        skip_stack.pop()

    if not root_seen:
        raise ValueError(f"SBML document is empty: {source_name}")
    if model_element is None:
        raise ValueError(f"SBML document has no model: {source_name}")

    if fbc_seen:
        model.model_info["fbc_version"] = 2
    if groups_seen:
        model.model_info["groups_version"] = 1

    # Resolve forward references after the streaming pass.
    for species_id, species in model.species.items():
        if species["compartment"] not in model.compartments:
            raise ValueError(f"Species {species_id!r} references unknown compartment {species['compartment']!r}")
    for reaction_id, reaction in model.reactions.items():
        referenced_species = (
            reaction["reactants"].keys()
            | reaction["products"].keys()
            | {modifier["species"] for modifier in reaction["modifiers"]}
        )
        for species_id in referenced_species:
            if species_id not in model.species:
                raise ValueError(f"Reaction {reaction_id!r} references unknown species {species_id!r}")
        for species_id in reaction["reactants"].keys() | reaction["products"].keys():
            species = model.species[species_id]
            if species["constant"] and not species["boundary_condition"]:
                raise ValueError(f"Reaction {reaction_id!r} references constant non-boundary species {species_id!r}")
        for side in ("lower", "upper"):
            ref = reaction[f"{side}_bound_ref"]
            if ref is None:
                raise ValueError(f"Reaction {reaction_id!r} is missing FBC {side}FluxBound")
            parameter = model.parameters.get(ref)
            if parameter is None:
                raise ValueError(f"Reaction {reaction_id!r} has unresolved {side}FluxBound parameter {ref!r}")
            if parameter["value"] is None:
                raise ValueError(f"Flux-bound parameter {ref!r} has no value")
            if not parameter["constant"]:
                raise ValueError(f"Flux-bound parameter {ref!r} is not constant")
            reaction[f"{side}_bound"] = parameter["value"]
        if reaction["lower_bound"] > reaction["upper_bound"]:
            raise ValueError(f"Reaction {reaction_id!r} has lower bound above upper bound")
        if (
            model.model_info.get("fbc_strict") is True
            and reaction["reversible"] is False
            and reaction["lower_bound"] is not None
            and reaction["lower_bound"] < 0
        ):
            raise ValueError(f"Irreversible reaction {reaction_id!r} has a negative FBC lower bound")
        for gene_id in reaction["gpr_genes"]:
            if gene_id not in model.gene_products:
                raise ValueError(f"Reaction {reaction_id!r} GPR references unknown gene product {gene_id!r}")
        reaction.pop("gpr_genes")
    for gene_id, gene in model.gene_products.items():
        species_id = gene["associated_species"]
        if species_id is not None and species_id not in model.species:
            raise ValueError(f"Gene product {gene_id!r} references unknown associated species {species_id!r}")
    for objective_id, objective in model.objectives.items():
        for reaction_id in objective["coefficients"]:
            if reaction_id not in model.reactions:
                raise ValueError(f"Objective {objective_id!r} references unknown reaction {reaction_id!r}")
    if model.active_objective is not None and model.active_objective not in model.objectives:
        raise ValueError(f"Unknown active FBC objective {model.active_objective!r}")
    for group_id, group in model.groups.items():
        for member in group["members"]:
            ref = member["id_ref"]
            meta_ref = member["meta_id_ref"]
            if ref is not None and ref not in ids and ref not in model.unit_definitions:
                raise ValueError(f"Group {group_id!r} references unknown id {ref!r}")
            if meta_ref is not None and meta_ref not in meta_ids:
                raise ValueError(f"Group {group_id!r} references unknown metaid {meta_ref!r}")
    return model


def read_sbml(
    source: str | os.PathLike | BinaryIO,
    *,
    annotations: bool = False,
    compression: str | None = "auto",
    timeout: float = 30.0,
) -> MetabolicModel:
    """Read supported metabolic SBML into plain Python records.

    Supports SBML Level 3 Version 1 Core, FBC Version 2, and optional Groups
    Version 1. Inputs can be local paths, HTTP(S) URLs, or binary readable
    streams. Compression is detected from magic bytes by default. Unsupported
    dynamic semantics, missing stoichiometry, and unresolved or incomplete
    flux bounds raise ``ValueError``.

    This parser uses only the Python standard library. Boundary and constant
    species remain in ``model.species``. With ``annotations=True``, embedded
    RDF resource URIs are stored on their owning record; annotation URIs are not fetched.

    Args:
        source: SBML path, HTTP(S) URL, or binary readable stream. Caller-owned
            streams are read from their current position and remain open.
        annotations: Include embedded RDF resource URIs in record metadata.
        compression: ``"auto"`` detects gzip, bz2, or xz by magic bytes;
            ``None`` disables decompression, and an explicit codec forces it.
        timeout: HTTP request timeout in seconds.

    Returns:
        A :class:`MetabolicModel` containing the parsed SBML records.

    Raises:
        ValueError: If the SBML document uses unsupported semantics or
            contains invalid or unresolved references.
        xml.etree.ElementTree.ParseError: If the XML is not well formed.
    """
    source_name = str(getattr(source, "name", source))
    with _open_binary(source, compression=compression, timeout=timeout) as stream:
        return _read_model(stream, annotations=annotations, source_name=source_name)

import abc
import numpy as np
from corneto._io import load_sif
from typing import Optional, Iterable, Set, Union, Dict, List
from corneto._typing import StrOrInt, TupleSIF
from corneto._constants import *


class Properties:
    def __init__(
        self,
        renet: "ReNet",
        species_values: Optional[Dict[int, float]] = None,
        reaction_values: Optional[Dict[int, float]] = None,
    ) -> None:
        self._renet = renet
        if reaction_values is None:
            reaction_values = {}
        if species_values is None:
            species_values = {}
        self._reaction_values = reaction_values
        self._species_values = species_values

    def reaction_value(self, reaction: StrOrInt, default: float = 0) -> float:
        if isinstance(reaction, str):
            reaction = self._renet.get_reaction_id(reaction)
        return self._reaction_values.get(reaction, default)

    def reaction_values(
        self, reactions: Optional[Iterable[int]] = None, default: float = 0
    ) -> List[float]:
        if reactions is None:
            reactions = range(len(self._renet.reactions))
        return [self.reaction_value(r, default) for r in reactions]

    def species_value(self, species: StrOrInt, default: float = 0) -> float:
        spid: int
        if isinstance(species, str):
            spid = self._renet.get_species_id(species)
        elif isinstance(species, int):
            spid = species
        else:
            raise ValueError(f"Invalid species: {species}")
        return self._species_values.get(spid, default)

    def species_values(
        self, species: Optional[Iterable[int]] = None, default: float = 0
    ) -> List[float]:
        if species is None:
            species = range(len(self._renet.species))
        return [self.species_value(s, default) for s in species]

    def copy(self):
        return Properties(
            self._renet, self._species_values.copy(), self._reaction_values.copy()
        )

    def select(
        self, species: Iterable[StrOrInt], reactions: Iterable[StrOrInt]
    ) -> "Properties":
        return Properties(
            self._renet,
            {i: self.species_value(s) for i, s in enumerate(species)},  # type: ignore
            {i: self.reaction_value(r) for i, r in enumerate(reactions)},  # type: ignore
        )


class ReNet(abc.ABC):
    def __init__(
        self, species: List[str], reactions: List[str], indexed: bool = True
    ) -> None:
        super().__init__()
        self._species = species
        self._reactions = reactions
        self._indexed = indexed
        if indexed:
            self._reaction_index = {r: i for i, r in enumerate(reactions)}
            self._species_index = {s: i for i, s in enumerate(species)}
        self.properties = Properties(self)

    @property
    def species(self):
        return self._species

    @property
    def reactions(self):
        return self._reactions

    @property
    def num_species(self) -> int:
        return len(self._species)

    @property
    def num_reactions(self) -> int:
        return len(self._reactions)

    @property
    def stoichiometry(self):
        return self.get_stoichiometry()

    @abc.abstractmethod
    def get_stoichiometry(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_reactants_of_reaction(self, reaction_id: int) -> Set[int]:
        pass

    @abc.abstractmethod
    def get_products_of_reaction(self, reaction_id: int) -> Set[int]:
        pass

    @abc.abstractmethod
    def get_reactions_with_product(self, species_id: int) -> Set[int]:
        pass

    @abc.abstractmethod
    def get_reactions_with_reactant(self, species_id: int) -> Set[int]:
        pass

    @abc.abstractmethod
    def _select_reactions(self, reaction_ids: List[int]) -> "ReNet":
        pass

    @abc.abstractmethod
    def _add_reaction(self, coeffs: Dict[str, int]):
        pass

    @abc.abstractmethod
    def _add_species(self, names: List[str]) -> None:
        pass

    def add_species(
        self,
        names: List[str],
        values: Optional[Dict[str, float]] = None,
        inplace: bool = True,
    ) -> "ReNet":
        if not inplace:
            n = self.copy()
        else:
            n = self
        for name in names:
            if name in n.species:
                raise ValueError(f"Species {name} already exists")
            if n._indexed:
                n._species.append(name)
                n._species_index[name] = len(n._species) - 1
            else:
                n._species.append(name)
        n._add_species(names)
        if values is not None:
            for name, value in values.items():
                n.properties._species_values[n.get_species_id(name)] = value
        return n

    def add_reaction(
        self,
        name: str,
        coeffs: Dict[str, int],
        value: Optional[float] = None,
        inplace: bool = True,
    ) -> "ReNet":
        if not inplace:
            n = self.copy()
        else:
            n = self
        if name in n.reactions:
            raise ValueError(f"Reaction {name} already exists")
        n._reactions.append(name)
        if n._indexed:
            n._reaction_index[name] = len(n._reactions) - 1
        n._add_reaction(coeffs)
        if value is not None:
            n.properties._reaction_values[n.get_reaction_id(name)] = value
        return n

    def add_reactions(
        self,
        reactions: Dict[str, Dict[str, int]],
        values: Dict[str, float] = dict(),
        inplace: bool = True,
    ) -> "ReNet":
        if not inplace:
            n = self.copy()
        else:
            n = self
        for k, v in reactions.items():
            n.add_reaction(k, v, value=values.get(k, None))
        return n

    def select_reactions(self, ids: Iterable[int], neighborhood=0) -> "ReNet":
        reaction_ids = set(ids)
        for _ in range(neighborhood):
            reactant_ids = self.get_reactants(reaction_ids)
            product_ids = self.get_products(reaction_ids)
            species_ids = reactant_ids | product_ids
            reaction_ids |= self.get_reactions(species_ids, species_ids)
        return self._select_reactions(list(reaction_ids))

    def select_species(
        self,
        reactant_ids: Optional[Iterable[int]] = None,
        product_ids: Optional[Iterable[int]] = None,
        union: bool = True,
        neighborhood=0,
    ) -> "ReNet":
        if reactant_ids is None and product_ids is None:
            if neighborhood > 0:
                raise ValueError(
                    "At least one of reactant_ids or product_ids must be specified"
                )
            return self
        rids: Set[int] = set()
        pids: Set[int] = set()
        if reactant_ids is None:
            rids = set()
        else:
            rids = set(reactant_ids)
        if product_ids is None:
            pids = set()
        else:
            pids = set(product_ids)
        for _ in range(neighborhood + 1):
            reaction_ids = self.get_reactions(
                reactant_ids=rids, product_ids=pids, union=union
            )
            rids |= self.get_reactants(rids)
            pids |= self.get_products(pids)
        return self.select_reactions(reaction_ids)

    def select(
        self,
        ids: Union[StrOrInt, List[StrOrInt]],
        id_type: IdType = IdType.SPECIES,
        neighborhood: int = 0,
    ) -> "ReNet":
        nids: List[int] = []
        if isinstance(ids, int) or isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            raise ValueError(f"ids must be a list of ints or strings")
        for id in ids:
            if isinstance(id, int):
                nids.append(id)
            elif isinstance(id, str):
                if id_type == IdType.SPECIES:
                    nids.append(self.get_species_id(id))
                elif id_type == IdType.REACTION:
                    nids.append(self.get_reaction_id(id))
                else:
                    raise ValueError(f"Invalid id_type: {id_type}")
            else:
                raise ValueError(f"Invalid id: {id}")
        if id_type == IdType.SPECIES:
            return self.select_species(nids, nids, neighborhood=neighborhood)
        else:
            return self.select_reactions(nids, neighborhood)

    @abc.abstractmethod
    def copy(self):
        pass

    def get_reaction_id(self, reaction_name: str) -> int:
        if self._indexed:
            return self._reaction_index[reaction_name]
        return self._reactions.index(reaction_name)

    def get_reaction_ids(self, reaction_names: Iterable[str]) -> List[int]:
        return [self.get_reaction_id(r) for r in reaction_names]

    def get_species_id(self, species_name: str):
        if self._indexed:
            return self._species_index[species_name]
        return self._species.index(species_name)

    def get_species_ids(self, species_names: Iterable[str]) -> List[int]:
        return [self.get_species_id(s) for s in species_names]

    def get_reactants(self, reaction_ids: Optional[Iterable[int]] = None) -> Set[int]:
        reactant_ids = set()
        if reaction_ids is None:
            reaction_ids = range(len(self._reactions))
        for reaction_id in reaction_ids:
            reactant_ids |= self.get_reactants_of_reaction(reaction_id)
        return reactant_ids

    def get_products(self, reaction_ids: Optional[Iterable[int]] = None) -> Set[int]:
        product_ids = set()
        if reaction_ids is None:
            reaction_ids = range(len(self._reactions))
        for reaction_id in reaction_ids:
            product_ids |= self.get_products_of_reaction(reaction_id)
        return product_ids

    def get_reactions_with_products(self, species_ids: Iterable[int]):
        reaction_ids = set()
        for species_id in species_ids:
            reaction_ids |= self.get_reactions_with_product(species_id)
        return reaction_ids

    def get_reactions_with_reactants(self, species_ids: Iterable[int]):
        reaction_ids = set()
        for species_id in species_ids:
            reaction_ids |= self.get_reactions_with_reactant(species_id)
        return reaction_ids

    def get_reactions(
        self,
        reactant_ids: Optional[Union[int, Iterable[int]]] = None,
        product_ids: Optional[Union[int, Iterable[int]]] = None,
        union: bool = True,
    ):
        if isinstance(reactant_ids, int):
            reactant_ids = [reactant_ids]
        if isinstance(product_ids, int):
            product_ids = [product_ids]
        r1, r2 = set(), set()
        if product_ids is not None:
            r1 = self.get_reactions_with_products(product_ids)
        if reactant_ids is not None:
            r2 = self.get_reactions_with_reactants(reactant_ids)
        if product_ids is None and reactant_ids is None:
            return {i for i in range(len(self._reactions))}
        if union or r1 is None or r2 is None:
            return r1 | r2
        else:
            return r1 & r2

    def reaction_names(self, reaction_ids: Iterable[int]) -> List[str]:
        return [self.reactions[i] for i in reaction_ids]

    def species_names(self, species_ids: Iterable[int]) -> List[str]:
        return [self.species[i] for i in species_ids]

    def get_ids(
        self, names: Union[StrOrInt, List[StrOrInt]], id_type: IdType = IdType.REACTION
    ) -> List[int]:
        if isinstance(names, int) or isinstance(names, str):
            names = [names]
        if id_type == IdType.REACTION:
            return [self.get_reaction_id(n) if isinstance(n, str) else n for n in names]
        else:
            return [self.get_species_id(n) if isinstance(n, str) else n for n in names]

    def successors(
        self, ids: Iterable[int], id_type: IdType = IdType.REACTION, rev: bool = False
    ) -> Set[int]:
        if id_type == IdType.REACTION:
            if not rev:
                return self.get_reactions_with_reactants(self.get_products(ids))
            else:
                return self.get_reactions_with_products(self.get_reactants(ids))
        elif id_type == IdType.SPECIES:
            if not rev:
                return self.get_products(self.get_reactions_with_reactants(ids))
            else:
                return self.get_reactants(self.get_reactions_with_products(ids))
        else:
            raise ValueError("id_type must be either REACTION or SPECIES")

    def bfs(
        self, ids: List[StrOrInt], id_type: IdType = IdType.SPECIES, rev: bool = False
    ) -> Dict[int, int]:
        nids = self.get_ids(ids, id_type)
        layer = 0
        visited = {s: layer for s in nids}
        succ = self.successors(nids, id_type=id_type, rev=rev)
        while succ:
            layer += 1
            new = []
            for s in succ:
                l = visited.get(s, np.inf)
                if layer < l:
                    visited[s] = layer
                    new.append(s)
            succ = self.successors(new, id_type=id_type, rev=rev)
        return visited

    def prune(
        self,
        source: List[StrOrInt],
        target: List[StrOrInt],
        id_type: IdType = IdType.SPECIES,
    ) -> "ReNet":
        forward = set(self.bfs(source, id_type=id_type).keys())
        backward = set(self.bfs(target, id_type=id_type, rev=True).keys())
        reachable = list(forward.intersection(backward))
        reactions = self.get_reactions(
            reactant_ids=reachable, product_ids=reachable, union=False
        )
        return self.select_reactions(reactions)

    @staticmethod
    def from_sif(
        sif: Union[str, List[TupleSIF]],
        delimiter: str = "\t",
        has_header: bool = False,
        discard_self_loops: Optional[bool] = True,
        sparse=False,
    ) -> "ReNet":
        if sparse:
            # TODO: Add SparseReNet implementation
            raise NotImplementedError("Sparse matrices not yet implemented")
        if isinstance(sif, str):
            return ReNet.from_sif_file(
                sif, delimiter, has_header, discard_self_loops, sparse
            )
        elif isinstance(sif, list):
            return ReNet.from_sif_list(sif, sparse)
        else:
            raise ValueError("sif must be either a string or a list of tuples")

    @staticmethod
    def from_sif_file(
        sif_file: str,
        delimiter: str = "\t",
        has_header: bool = False,
        discard_self_loops: Optional[bool] = True,
        sparse=False,
    ) -> "ReNet":
        if sparse:
            # TODO: Add SparseReNet implementation
            raise NotImplementedError("Sparse matrices not yet implemented")

        S, s, r, p = load_sif(sif_file, delimiter, has_header, discard_self_loops)
        renet = DenseReNet(S, s, r)
        renet.properties._reaction_values = p
        return renet

    @staticmethod
    def from_sif_list(tpl: List[TupleSIF], sparse=False) -> "ReNet":
        from corneto._io import load_sif_from_tuples

        if sparse:
            # TODO: Add SparseReNet implementation
            raise NotImplementedError("Sparse matrices not yet implemented")

        S, s, r, p = load_sif_from_tuples(tpl)
        renet = DenseReNet(S, s, r)
        renet.properties._reaction_values = p
        return renet

    @staticmethod
    def create(
        stoichiometry: np.ndarray,
        species: Optional[List[str]] = None,
        reactions: Optional[List[str]] = None,
    ) -> "ReNet":
        if species is None:
            species = [f"S_{i}" for i in range(stoichiometry.shape[0])]
        if reactions is None:
            reactions = [f"R_{i}" for i in range(stoichiometry.shape[1])]
        return DenseReNet(stoichiometry, species, reactions)

    def nxgraph(self, reactions: Optional[Iterable[StrOrInt]] = None):
        from corneto._nx import to_nxgraph

        return to_nxgraph(self, reactions)

    def plot(self, **kwargs):
        from corneto._nx import plot

        return plot(self.nxgraph(), **kwargs)


class DenseReNet(ReNet):
    def __init__(
        self, stoichiometry: np.ndarray, species: List[str], reactions: List[str]
    ) -> None:
        super().__init__(species, reactions)
        self._stoichiometry = stoichiometry

    def get_stoichiometry(self) -> np.ndarray:
        return self._stoichiometry

    def get_reactants_of_reaction(self, reaction_id: int) -> Set[int]:
        return set(np.where(self._stoichiometry[:, reaction_id] < 0)[0])

    def get_products_of_reaction(self, reaction_id: int) -> Set[int]:
        return set(np.where(self._stoichiometry[:, reaction_id] > 0)[0])

    def get_reactions_with_product(self, species_id: int) -> Set[int]:
        return set(np.where(self._stoichiometry[species_id, :] > 0)[0])

    def get_reactions_with_reactant(self, species_id: int) -> Set[int]:
        return set(np.where(self._stoichiometry[species_id, :] < 0)[0])

    def _select_reactions(self, reaction_ids: List[int]) -> "ReNet":
        S = self._stoichiometry[:, reaction_ids]
        non_empty = np.sum(np.abs(S), axis=1) > 0
        species_ids = np.where(non_empty)[0]
        species = [self.species[i] for i in species_ids]
        reactions = [self.reactions[i] for i in reaction_ids]
        rn = DenseReNet(S[species_ids, :], species, reactions)
        rn.properties = self.properties.select(species, reactions)
        rn.properties._renet = rn
        return rn

    def copy(self):
        renet = DenseReNet(
            self._stoichiometry.copy(), self._species.copy(), self._reactions.copy()
        )
        renet.properties = self.properties.copy()
        renet.properties._renet = renet
        return renet

    def _add_species(self, names: List[str]) -> None:
        v = self._stoichiometry.shape[1] if len(self._stoichiometry.shape) >= 2 else 1
        rows = np.zeros((len(names), v))
        self._stoichiometry = (
            np.vstack((self._stoichiometry, rows)) if self._stoichiometry.size else rows
        )

    def _add_reaction(self, coeffs: Dict[str, int]):
        new_species = list(set(coeffs.keys()) - set(self.species))
        st_sz = self._stoichiometry.size
        if len(new_species) > 0:
            self.add_species(new_species)
        # Add new column to the stoichiometric matrix if it was non-empty
        if st_sz > 0:
            col = np.zeros((self._stoichiometry.shape[0], 1))
            self._stoichiometry = np.hstack((self._stoichiometry, col))
        # Add the coefficients
        for s, coeff in coeffs.items():
            self._stoichiometry[self.get_species_id(s), -1] = coeff

    @staticmethod
    def empty():
        return DenseReNet(np.array([]), [], [])

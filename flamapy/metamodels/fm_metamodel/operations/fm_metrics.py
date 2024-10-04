from typing import Any, Callable, Optional, cast
from collections import defaultdict
import statistics

from flamapy.core.models.variability_model import VariabilityModel
from flamapy.core.operations.metrics_operation import Metrics
from flamapy.metamodels.fm_metamodel.models import FeatureModel, Feature


def metric_method(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Decorator to mark a method as a metric method.
    It has the value of the measure, it can also have a size and a ratio.
    Example:
        property name: Abstract Features.
        description: The description of the property
        value (optional): the list of abstract features.
        size (optional): the length of the list.
        ratio (optional): the percentage of abstract features with regards the total
        number of features.
    """
    if not hasattr(func, "_is_metric_method"):
        setattr(func, "_is_metric_method", True)
    return func


class FMMetrics(Metrics):  # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[FeatureModel] = None
        self.result: list[dict[str, Any]] = []
        self._model_type_extension = "fm"
        self._features = None
        self._abstract_features = None
        self._concrete_features = None
        self._tree_relationships = None
        self._root_feature = None
        self._feature_groups = None
        self._branching_factor = None
        self._depth_of_tree = None
        self._cross_tree_constraints = None
        self._simple_constraints = None
        self._complex_constraints = None
        self._feature_attributes = None
        self._features_in_constraints = None

    @property
    def model_type_extension(self) -> str:
        return self._model_type_extension

    @model_type_extension.setter
    def model_type_extension(self, ext: str) -> None:
        self._model_type_extension = ext

    def get_result(self) -> list[dict[str, Any]]:
        return self.result

    def calculate_metamodel_metrics(self, model: VariabilityModel) -> list[dict[str, Any]]:
        self.model = cast(FeatureModel, model)

        # Calculate all metrics in a single traverse of the feature tree
        self._metrics = traverse_metrics(self.model)

        # Get all methods that are marked with the metric_method decorator
        metric_methods = [
            getattr(self, method_name)
            for method_name in dir(self)
            if callable(getattr(self, method_name))
            and hasattr(getattr(self, method_name), "_is_metric_method")
        ]

        if self.filter is not None:
            metric_methods = [
                method for method in metric_methods if method.__name__ in self.filter
            ]

        return [method() for method in metric_methods]

    @staticmethod
    def get_ratio(value1: Any, value2: Any, precision: int = 4) -> float:
        if value2 == 0:
            return 0.0
        return float(round(value1 / value2, precision))
    
    @metric_method
    def abstract_features(self) -> dict[str, Any]:
        """Features used to structure the feature model that, however, do not have any
        impact at implementation level."""
        self._abstract_features = Metric(name='Abstract features', 
                                         doc=self.abstract_features.__doc__, 
                                         result=self._metrics['Abstract features'],
                                         ratio=self.get_ratio(self._metrics['Abstract features'], 
                                                              self._metrics['Features']),
                                         parent=self._features)
        return self._abstract_features.to_json()

    @metric_method
    def concrete_features(self) -> dict[str, Any]:
        """Features that are mapped to at least one implementation artifact."""
        self._concrete_features = Metric(name='Concrete features', 
                                         doc=self.concrete_features.__doc__, 
                                         result=self._metrics['Concrete features'],
                                         ratio=self.get_ratio(self._metrics['Concrete features'], 
                                                              self._metrics['Features']),
                                         parent=self._features)
        return self._concrete_features.to_json()

    @metric_method
    def leaf_features(self) -> dict[str, Any]:
        """Features that have not subfeatures (aka 'primitive features' or 'terminal features')."""
        return Metric(name='Leaf features', 
                      doc=self.leaf_features.__doc__, 
                      result=self._metrics['Leaf features'],
                      ratio=self.get_ratio(self._metrics['Leaf features'], 
                                           self._metrics['Features']),
                      parent=self._features).to_json()

    @metric_method
    def compound_features(self) -> dict[str, Any]:
        """Features that have subfeatures."""
        return Metric(name='Compound features', 
                      doc=self.compound_features.__doc__, 
                      result=self._metrics['Compound features'],
                      ratio=self.get_ratio(self._metrics['Compound features'], 
                                           self._metrics['Features']),
                      parent=self._features).to_json()

    @metric_method
    def concrete_compound_features(self) -> dict[str, Any]:
        """Concrete and compound features."""
        return Metric(name='Concrete compound features', 
                      doc=self.concrete_compound_features.__doc__, 
                      result=self._metrics['Concrete compound features'],
                      ratio=self.get_ratio(self._metrics['Concrete compound features'], 
                                           self._metrics['Concrete features']),
                      parent=self._concrete_features).to_json()

    @metric_method
    def concrete_leaf_features(self) -> dict[str, Any]:
        """Concrete and leaf features."""
        return Metric(name='Concrete leaf features', 
                      doc=self.concrete_leaf_features.__doc__, 
                      result=self._metrics['Concrete leaf features'],
                      ratio=self.get_ratio(self._metrics['Concrete leaf features'], 
                                           self._metrics['Concrete features']),
                      parent=self._concrete_features).to_json()

    @metric_method
    def abstract_compound_features(self) -> dict[str, Any]:
        """Abstract and compound features."""
        return Metric(name='Abstract compound features', 
                      doc=self.abstract_compound_features.__doc__, 
                      result=self._metrics['Abstract compound features'],
                      ratio=self.get_ratio(self._metrics['Abstract compound features'], 
                                           self._metrics['Abstract features']),
                      parent=self._abstract_features).to_json()

    @metric_method
    def abstract_leaf_features(self) -> dict[str, Any]:
        """Abstract and leaf features."""
        return Metric(name='Abstract leaf features', 
                      doc=self.abstract_leaf_features.__doc__, 
                      result=self._metrics['Abstract leaf features'],
                      ratio=self.get_ratio(self._metrics['Abstract leaf features'], 
                                           self._metrics['Abstract features']),
                      parent=self._abstract_features).to_json()

    @metric_method
    def tree_relationships(self) -> dict[str, Any]:
        """Number of relationships (edges) of the feature model."""
        self._tree_relationships = Metric(name='Tree relationships', 
                                          doc=self.tree_relationships.__doc__, 
                                          result=self._metrics['Tree relationships'])
        return self._tree_relationships.to_json()

    @metric_method
    def root_feature(self) -> dict[str, Any]:
        """The root of the feature model."""
        self._root_feature = Metric(name='Root feature', 
                                    doc=self.root_feature.__doc__, 
                                    result=self._metrics['Root feature'],
                                    ratio=self.get_ratio(self._metrics['Root feature'], 
                                                         self._metrics['Features']),
                                    parent=self._features)
        return self._root_feature.to_json()

    @metric_method
    def top_features(self) -> dict[str, Any]:
        """Features that are first descendants of the root."""
        return Metric(name='Top feature', 
                      doc=self.top_features.__doc__, 
                      result=self._metrics['Top feature'],
                      ratio=self.get_ratio(self._metrics['Top feature'], 
                                           self._metrics['Features']),
                      parent=self._root_feature).to_json()

    @metric_method
    def solitary_features(self) -> dict[str, Any]:
        """Features that are not grouped in a feature group."""
        return Metric(name='Solitary features', 
                      doc=self.solitary_features.__doc__, 
                      result=self._metrics['Solitary features'],
                      ratio=self.get_ratio(self._metrics['Solitary features'], 
                                           self._metrics['Features']),
                      parent=self._features).to_json()

    @metric_method
    def grouped_features(self) -> dict[str, Any]:
        """Features that occurs in a feature group."""
        return Metric(name='Grouped features', 
                      doc=self.grouped_features.__doc__, 
                      result=self._metrics['Grouped features'],
                      ratio=self.get_ratio(self._metrics['Grouped features'], 
                                           self._metrics['Features']),
                      parent=self._features).to_json()

    @metric_method
    def mandatory_features(self) -> dict[str, Any]:
        """Features marked as mandatory that need to be selected if its parent is selected."""
        return Metric(name='Mandatory features', 
                      doc=self.mandatory_features.__doc__, 
                      result=self._metrics['Mandatory features'],
                      ratio=self.get_ratio(self._metrics['Mandatory features'], 
                                           self._metrics['Tree relationships']),
                      parent=self._tree_relationships).to_json()

    @metric_method
    def optional_features(self) -> dict[str, Any]:
        """Feature marked as optional."""
        return Metric(name='Optional features', 
                      doc=self.optional_features.__doc__, 
                      result=self._metrics['Optional features'],
                      ratio=self.get_ratio(self._metrics['Optional features'], 
                                           self._metrics['Tree relationships']),
                      parent=self._tree_relationships).to_json()

    @metric_method
    def feature_groups(self) -> dict[str, Any]:
        """Features that express a choice over the grouped features in a group."""
        self._feature_groups = Metric(name='Feature groups', 
                                      doc=self.feature_groups.__doc__, 
                                      result=self._metrics['Feature groups'],
                                      ratio=self.get_ratio(self._metrics['Feature groups'], 
                                                           self._metrics['Tree relationships']),
                                      parent=self._tree_relationships)
        return self._feature_groups.to_json()

    @metric_method
    def alternative_groups(self) -> dict[str, Any]:
        """Feature groups that require the selection of just one child (i.e., [1..1]
        cardinality)."""
        return Metric(name='Alternative groups', 
                      doc=self.alternative_groups.__doc__, 
                      result=self._metrics['Alternative groups'],
                      ratio=self.get_ratio(self._metrics['Alternative groups'], 
                                           self._metrics['Feature groups']),
                      parent=self._feature_groups).to_json()

    @metric_method
    def or_groups(self) -> dict[str, Any]:
        """Feature groups that require the selection of at least one child (i.e., [1..*]
        cardinality)."""
        return Metric(name='Or groups', 
                      doc=self.or_groups.__doc__, 
                      result=self._metrics['Or groups'],
                      ratio=self.get_ratio(self._metrics['Or groups'], 
                                           self._metrics['Feature groups']),
                      parent=self._feature_groups).to_json()

    @metric_method
    def mutex_groups(self) -> dict[str, Any]:
        """Feature groups that require the selection of zero or just one child (i.e.,
        [0..1] cardinality)."""
        return Metric(name='Mutex groups', 
                      doc=self.mutex_groups.__doc__, 
                      result=self._metrics['Mutex groups'],
                      ratio=self.get_ratio(self._metrics['Mutex groups'], 
                                           self._metrics['Feature groups']),
                      parent=self._feature_groups).to_json()

    @metric_method
    def cardinality_groups(self) -> dict[str, Any]:
        """Feature groups with arbitrary cardinality [a..b] that require the selection
        of a minimum and a maximum number of children."""
        return Metric(name='Cardinality groups', 
                      doc=self.cardinality_groups.__doc__, 
                      result=self._metrics['Cardinality groups'],
                      ratio=self.get_ratio(self._metrics['Cardinality groups'], 
                                           self._metrics['Feature groups']),
                      parent=self._feature_groups).to_json()

    @metric_method
    def branching_factor(self) -> dict[str, Any]:
        """Average number of children per non-leaf feature (aka 'Ratio of Variability')."""
        self._branching_factor = Metric(name='Branching factor', 
                                        doc=self.branching_factor.__doc__, 
                                        result=self._metrics['Branching factor'])
        return self._branching_factor.to_json()

    @metric_method
    def min_children_per_feature(self) -> dict[str, Any]:
        """Minimal number of children per non-leaf feature."""
        return Metric(name='Min children per feature', 
                      doc=self.min_children_per_feature.__doc__, 
                      result=self._metrics['Min children per feature'],
                      ratio=self.get_ratio(self._metrics['Min children per feature'], 
                                           self._metrics['Branching factor']),
                      parent=self._branching_factor).to_json()

    @metric_method
    def max_children_per_feature(self) -> dict[str, Any]:
        """Maximal number of children per feature."""
        return Metric(name='Max children per feature', 
                      doc=self.max_children_per_feature.__doc__, 
                      result=self._metrics['Max children per feature'],
                      ratio=self.get_ratio(self._metrics['Max children per feature'], 
                                           self._metrics['Branching factor']),
                      parent=self._branching_factor).to_json()

    @metric_method
    def avg_children_per_feature(self) -> dict[str, Any]:
        """Average number of children per feature."""
        return Metric(name='Avg children per feature', 
                      doc=self.avg_children_per_feature.__doc__, 
                      result=self._metrics['Avg children per feature'],
                      ratio=self.get_ratio(self._metrics['Avg children per feature'], 
                                           self._metrics['Branching factor']),
                      parent=self._branching_factor).to_json()

    @metric_method
    def depth_tree(self) -> dict[str, Any]:
        """Number of features of the longest path from the root to the leaf features."""
        self._depth_of_tree = Metric(name='Depth of tree', 
                                     doc=self.depth_tree.__doc__, 
                                     result=self._metrics['Depth of tree'])
        return self._depth_of_tree.to_json()

    @metric_method
    def mean_depth_tree(self) -> dict[str, Any]:
        """Number of features of the mean path from the root to the leaf features."""
        return Metric(name='Mean depth of tree', 
                      doc=self.mean_depth_tree.__doc__, 
                      result=self._metrics['Mean depth of tree'],
                      parent=self._depth_of_tree).to_json()

    @metric_method
    def cross_tree_constraints(self) -> dict[str, Any]:
        """Textual cross-tree constraints."""
        self._cross_tree_constraints = Metric(name='Cross tree constraints', 
                                              doc=self.cross_tree_constraints.__doc__, 
                                              result=self._metrics['Cross tree constraints'])
        return self._cross_tree_constraints.to_json()

    @metric_method
    def single_feature_constraints(self) -> dict[str, Any]:
        """Constraints with a single feature or negated feature."""
        return Metric(name='Single feature constraints', 
                      doc=self.single_feature_constraints.__doc__, 
                      result=self._metrics['Single feature constraints'],
                      ratio=self.get_ratio(self._metrics['Single feature constraints'],
                                           self._metrics['Cross tree constraints']),
                      parent=self._cross_tree_constraints).to_json()

    @metric_method
    def simple_constraints(self) -> dict[str, Any]:
        """Requires and Excludes constraints."""
        self._simple_constraints = Metric(name='Simple constraints', 
                                         doc=self.simple_constraints.__doc__, 
                                         result=self._metrics['Simple constraints'],
                                         ratio=self.get_ratio(self._metrics['Simple constraints'], 
                                                              self._metrics['Cross tree constraints']),
                                         parent=self._cross_tree_constraints)
        return self._simple_constraints.to_json()

    @metric_method
    def requires_constraints(self) -> dict[str, Any]:
        """Constraints modeling that the activation of a feature f1 implies the
        activation of a feature f2."""
        return Metric(name='Requires constraints', 
                      doc=self.requires_constraints.__doc__, 
                      result=self._metrics['Requires constraints'],
                      ratio=self.get_ratio(self._metrics['Requires constraints'],
                                           self._metrics['Simple constraints']),
                      parent=self._simple_constraints).to_json()

    @metric_method
    def excludes_constraints(self) -> dict[str, Any]:
        """Constraints modeling that two features are mutually exclusive and cannot be
        activated together."""
        return Metric(name='Excludes constraints', 
                      doc=self.excludes_constraints.__doc__, 
                      result=self._metrics['Excludes constraints'],
                      ratio=self.get_ratio(self._metrics['Excludes constraints'],
                                           self._metrics['Simple constraints']),
                      parent=self._simple_constraints).to_json()

    @metric_method
    def complex_constraints(self) -> dict[str, Any]:
        """Constraints in arbitrary propositional logic formulae."""
        self._complex_constraints = Metric(name='Complex constraints', 
                                         doc=self.complex_constraints.__doc__, 
                                         result=self._metrics['Complex constraints'],
                                         ratio=self.get_ratio(self._metrics['Complex constraints'], 
                                                              self._metrics['Cross tree constraints']),
                                         parent=self._cross_tree_constraints)
        return self._complex_constraints.to_json()

    @metric_method
    def pseudo_complex_constraints(self) -> dict[str, Any]:
        """Constraints that are convertible to a set of simple constraints."""
        return Metric(name='Pseudo-complex constraints', 
                      doc=self.pseudo_complex_constraints.__doc__, 
                      result=self._metrics['Pseudo-complex constraints'],
                      ratio=self.get_ratio(self._metrics['Pseudo-complex constraints'],
                                           self._metrics['Complex constraints']),
                      parent=self._complex_constraints).to_json()

    @metric_method
    def strict_complex_constraints(self) -> dict[str, Any]:
        """Constraints that cannot be converted to a set of simple constraints."""
        return Metric(name='Strict-complex constraints', 
                      doc=self.strict_complex_constraints.__doc__, 
                      result=self._metrics['Strict-complex constraints'],
                      ratio=self.get_ratio(self._metrics['Strict-complex constraints'],
                                           self._metrics['Complex constraints']),
                      parent=self._complex_constraints).to_json()

    @metric_method
    def min_constraints_per_feature(self) -> dict[str, Any]:
        """The minimal number of constraints per feature."""
        return Metric(name='Min constraints per feature', 
                      doc=self.min_constraints_per_feature.__doc__, 
                      result=self._metrics['Min constraints per feature'],
                      parent=self._cross_tree_constraints).to_json()

    @metric_method
    def max_constraints_per_feature(self) -> dict[str, Any]:
        """The maximal number of constraints per feature."""
        return Metric(name='Max constraints per feature',
                      doc=self.max_constraints_per_feature.__doc__, 
                      result=self._metrics['Max constraints per feature'],
                      parent=self._cross_tree_constraints).to_json()

    @metric_method
    def avg_constraints_per_feature(self) -> dict[str, Any]:
        """The average number of constraints per feature."""
        return Metric(name='Avg constraints per feature', 
                      doc=self.avg_constraints_per_feature.__doc__, 
                      result=self._metrics['Avg constraints per feature'],
                      parent=self._cross_tree_constraints).to_json()

    @metric_method
    def features_in_constraints(self) -> dict[str, Any]:
        """Features involved in cross-tree constraints. The ratio to the total number of
        features is called 'Extra constraint representativeness (ECR)'."""
        self._features_in_constraints = Metric(name='Features in constraints', 
                                         doc=self.features_in_constraints.__doc__, 
                                         result=self._metrics['Features in constraints'],
                                         ratio=self.get_ratio(self._metrics['Features in constraints'], 
                                                              self._metrics['Features']),
                                         parent=self._cross_tree_constraints)
        return self._features_in_constraints.to_json()
    
    @metric_method
    def min_features_in_constraints(self) -> dict[str, Any]:
        """The minimal number of features involved in a cross-tree constraint."""
        return Metric(name='Min features in constraints', 
                      doc=self.min_features_in_constraints.__doc__, 
                      result=self._metrics['Min features in constraints'],
                      parent=self._features_in_constraints).to_json()
    
    @metric_method
    def max_features_in_constraints(self) -> dict[str, Any]:
        """The maximal number of features involved in a cross-tree constraint."""
        return Metric(name='Max features in constraints', 
                      doc=self.max_features_in_constraints.__doc__, 
                      result=self._metrics['Max features in constraints'],
                      parent=self._features_in_constraints).to_json()
    
    @metric_method
    def avg_features_in_constraints(self) -> dict[str, Any]:
        """The average number of features involved in a cross-tree constraint."""
        return Metric(name='Avg features in constraints', 
                      doc=self.avg_features_in_constraints.__doc__, 
                      result=self._metrics['Avg features in constraints'],
                      parent=self._features_in_constraints).to_json()

    @metric_method
    def feature_attributes(self) -> dict[str, Any]:
        """Features attributes in the model (i.e., number of distinct attributes)."""
        self._feature_attributes = Metric(name='Feature attributes', 
                                          doc=self.feature_attributes.__doc__, 
                                          result=self._metrics['Feature attributes'])
        return self._feature_attributes.to_json()

    @metric_method
    def features_with_attributes(self) -> dict[str, Any]:
        """Features that contain some attributes defined in the model."""
        return Metric(name='Features with attributes', 
                      doc=self.features_with_attributes.__doc__, 
                      result=self._metrics['Features with attributes'],
                      ratio=self.get_ratio(self._metrics['Features with attributes'], 
                                           self._metrics['Features']),
                      parent=self._feature_attributes).to_json()
    
    @metric_method
    def min_attributes_per_feature(self) -> dict[str, Any]:
        """The minimal number of attributes in a feature."""
        return Metric(name='Min attributes per feature', 
                      doc=self.min_attributes_per_feature.__doc__, 
                      result=self._metrics['Min attributes per feature'],
                      parent=self._feature_attributes).to_json()

    @metric_method
    def max_attributes_per_feature(self) -> dict[str, Any]:
        """The maximal number of attributes in a feature."""
        return Metric(name='Max attributes per feature', 
                      doc=self.max_attributes_per_feature.__doc__, 
                      result=self._metrics['Max attributes per feature'],
                      parent=self._feature_attributes).to_json()

    @metric_method
    def avg_attributes_per_feature(self) -> dict[str, Any]:
        """The average number of attributes in features with attributes."""
        return Metric(name='Avg attributes per feature', 
                      doc=self.avg_attributes_per_feature.__doc__, 
                      result=self._metrics['Avg attributes per feature'],
                      parent=self._feature_attributes).to_json()


class Metric:
    """Basic data that store a specific metric."""

    def __init__(self, 
                 name: str, 
                 doc: str, 
                 result: Any, 
                 ratio: Optional[float] = None,
                 parent: Optional['Metric'] = None,
                 level: Optional[int] = None) -> None:
        self.name = name
        self.doc = doc
        self.result = result
        self.ratio = ratio
        self.parent = parent
        self.level = level if level is not None else (0 if parent is None else parent.level + 1)

    def to_json(self) -> dict[str, Any]:
        return {'name': self.name,
                'documentation': self.doc,
                'result': self.result,
                'ratio': self.ratio,
                'parent': None if self.parent is None else self.parent.name,
                'level': self.level}


def traverse_metrics(fm: FeatureModel) -> dict[str, Any]:
    """Calculate all metrics from the feature model in only one traversing of the tree."""
    metrics: dict[str, int] = defaultdict(int)
    if fm is None:
        return metrics
    ## Features metrics
    metrics['Root feature'] = set()
    metrics['Mean depth of tree'] = []
    metrics['Branching factor'] = []
    metrics['Avg branching factor'] = []
    metrics['Feature attributes'] = set()
    metrics['Avg attributes per feature'] = []
    metrics['Avg attributes per feature with attributes'] = []
    metrics['Avg children per feature'] = []
    traverse_feature_metrics(fm.root, metrics)
    metrics['Root feature'] = len(metrics['Root feature'])
    metrics['Mean depth of tree'] = 0 if not metrics['Mean depth of tree'] else statistics.mean(metrics['Mean depth of tree'])
    metrics['Avg children per feature'] = 0 if not metrics['Avg children per feature'] else statistics.mean(metrics['Avg children per feature'])
    metrics['Branching factor'] = 0 if metrics['Branches'] == 0 else round(metrics['Children'] / metrics['Branches'], 2)
    metrics['Feature attributes'] = len(metrics['Feature attributes'])
    metrics['Min attributes per feature'] = 0 if not metrics['Avg attributes per feature'] else min(metrics['Avg attributes per feature'])
    metrics['Max attributes per feature'] = 0 if not metrics['Avg attributes per feature'] else max(metrics['Avg attributes per feature'])
    metrics['Avg attributes per feature'] = 0 if not metrics['Avg attributes per feature with attributes'] else statistics.mean(metrics['Avg attributes per feature with attributes'])
    ## Constraints metrics
    metrics['Features in constraints'] = set()
    metrics['Avg feature in constraints'] = []
    metrics['Avg constraints per feature'] = defaultdict(int)
    traverse_constraints_metrics(fm, metrics)
    metrics['Cross tree constraints'] = len(fm.get_constraints())
    metrics['Features in constraints'] = len(metrics['Features in constraints'])
    metrics['Min features in constraints'] = 0 if not metrics['Avg feature in constraints'] else min(metrics['Avg feature in constraints'])
    metrics['Max features in constraints'] = 0 if not metrics['Avg feature in constraints'] else max(metrics['Avg feature in constraints'])
    metrics['Avg feature in constraints'] = 0 if not metrics['Avg feature in constraints'] else statistics.mean(metrics['Avg feature in constraints'])
    metrics['Min constraints per feature'] = 0 if not metrics['Avg constraints per feature'] else min(metrics['Avg constraints per feature'].values())
    metrics['Max constraints per feature'] = 0 if not metrics['Avg constraints per feature'] else max(metrics['Avg constraints per feature'].values())
    metrics['Avg constraints per feature'] = 0 if not metrics['Avg constraints per feature'] else statistics.mean(metrics['Avg constraints per feature'].values())
    return metrics


def traverse_feature_metrics(feature: Feature, metrics: dict[str, Any], depth: int = 0) -> None:
    if feature is not None:
        metrics['Features'] += 1
        if feature.parent is None:
            metrics['Root feature'].add(feature.name)
        elif feature.parent.is_root():
            metrics['Top features'] += 1
        
        # Attributes
        attributes = feature.get_attributes()
        metrics['Avg attributes per feature'].append(len(attributes))
        if attributes:
            metrics['Feature with attributes'] += 1
            metrics['Avg attributes per feature with attributes'].append(len(attributes))
            for attribute in attributes:
                metrics['Feature attributes'].add(attribute.name)

        relations = feature.get_relations()
        n_children = 0
        if relations:  # it is a compound feature (non leaf)
            metrics['Compound features'] += 1
            if feature.is_abstract:
                metrics['Abstract features'] += 1
                metrics['Abstract compound features'] += 1
            else:
                metrics['Concrete features'] += 1
                metrics['Concrete compound features'] += 1
            metrics['Branches'] += 1

            for relation in feature.get_relations():
                metrics['Tree relationships'] += 1
                if relation.is_mandatory():
                    n_children += 1
                    metrics['Mandatory features'] += 1
                    metrics['Solitary features'] += 1
                    traverse_feature_metrics(relation.children[0], metrics, depth + 1)
                elif relation.is_optional():
                    n_children += 1
                    metrics['Optional features'] += 1
                    metrics['Solitary features'] += 1
                    traverse_feature_metrics(relation.children[0], metrics, depth + 1)
                elif relation.is_group():
                    n_children += len(relation.children)
                    metrics['Feature groups'] += 1
                    metrics['Grouped features'] += n_children
                    if relation.is_or():
                        metrics['Or groups'] += 1
                    elif relation.is_alternative():
                        metrics['Alternative groups'] += 1
                    elif relation.is_mutex():
                        metrics['Mutex groups'] += 1
                    else:
                        metrics['Cardinality groups'] += 1
                    for child in relation.children:
                        traverse_feature_metrics(child, metrics, depth + 1)
            metrics['Children'] += n_children
            metrics['Min children per features'] = min(metrics['Min children per features'], n_children)
            metrics['Max children per features'] = max(metrics['Max children per features'], n_children)
        else:  # it is a leaf feature
            metrics['Leaf features'] += 1
            if feature.is_abstract:
                metrics['Abstract features'] += 1
                metrics['Abstract leaf features'] += 1
            else:
                metrics['Concrete features'] += 1
                metrics['Concrete leaf features'] += 1
            metrics['Depth of tree'] = max(metrics['Depth of tree'], depth)
            metrics['Mean depth of tree'].append(depth)
        metrics['Avg children per feature'].append(n_children)


def traverse_constraints_metrics(fm: FeatureModel, metrics: dict[str, Any]) -> None:
    for ctc in fm.get_constraints():
        if ctc.is_single_feature_constraint():
            metrics['Single feature constraints'] += 1
        elif ctc.is_requires_constraint():
            metrics['Simple constraints'] += 1
            metrics['Requires constraints'] += 1
        elif ctc.is_excludes_constraint():
            metrics['Simple constraints'] += 1
            metrics['Excludes constraints'] += 1
        else:
            metrics['Complex constraints'] += 1
            if ctc.is_pseudocomplex_constraint():
                metrics['Pseudo-complex constraints'] += 1
            else:
                metrics['Strict-complex constraints'] += 1
        features = ctc.get_features()
        metrics['Features in constraints'].update(features)
        metrics['Avg feature in constraints'].append(len(features))
        for feature in features:
            metrics['Avg constraints per feature'][feature] += 1

    
METRICS_ORDER = [
    'Features',
    'Abstract features',
    'Abstract compound features',
    'Abstract leaf features',
    'Concrete features',
    'Concrete compound features',
    'Concrete leaf features',
    'Compound features',
    'Leaf features',
    'Root feature',
    'Top features',
    'Solitary features',
    'Grouped features',
    'Tree relationships',
    'Mandatory features',
    'Optional features',
    'Feature groups',
    'Alternative groups',
    'Or groups',
    'Mutex groups',
    'Cardinality groups',
    'Depth of tree',
    'Mean depth of tree',
    'Branching factor',
    'Min children per feature',
    'Max children per feature',
    'Avg children per feature',
    'Cross-tree constraints',
    'Single feature constraints',
    'Simple constraints',
    'Requires constraints',
    'Excludes constraints',
    'Complex constraints',
    'Pseudo-complex constraints',
    'Strict-complex constraints',
    'Features in constraints',
    'Min features in constraints',
    'Max features in constraints',
    'Avg features in constraints',
    'Min constraints per feature',
    'Max constraints per feature',
    'Avg constraints per feature',
    'Feature attributes',
    'Features with attributes',
    'Min attributes per feature',
    'Max attributes per feature',
    'Avg attributes per feature',
]

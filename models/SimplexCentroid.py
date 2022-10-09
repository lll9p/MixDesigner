#!/usr/bin/env python
# coding: utf-8
from decimal import Decimal
from fractions import Fraction
from itertools import chain
from logging import error
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

import numpy as np
import openpyxl
import openpyxl.styles
from typing_extensions import Self

from .BaseModel import BaseModel
from .types import (BoundsType, ComponentNamesType, DefaultsType, PointsType,
                    TargetNamesType, TestPointType)

ModelType = TypeVar("ModelType", bound="SimplexCentroid")

LIMIT_DENOMINATOR = 1000000


def np_formatter_func(x: Any) -> str:
    if isinstance(x, Fraction):
        return str(x.limit_denominator(LIMIT_DENOMINATOR))
    else:
        return str(x)


np.set_printoptions(formatter={"all": np_formatter_func})


class SimplexCentroid(BaseModel):
    """
    SimplexCentroid model for mixdesiner.
    Theory
    ------
    Convert proportion to projected space, fit the response surface coefficients.
    Experiments numbers equals to power(n_components,2)-1.
    Let n=n_components, en=2**n-1
    Powerset mask is (Removed first empty subset.)
    """

    def __init__(self, defaults: Optional[DefaultsType] = None) -> None:
        """
        Initial parameters for this estimator.

        Parameters
        ----------
        defaults : dict | None, default=None
            Specific the defaults of model.
            If it's a dict of params, use them to build model.
            If it's true of something not false or none, build with 3 components model,
            with 0 0 0 lb and 1 1 1 ub, with 3 test points.
            If it's none or false, build an empty model, then use `with_` methods to complete build.

        Returns
        -------
        None
        """
        super().__init__(name="SimplexCentroid")
        self.built = False
        if isinstance(defaults, dict):
            n_components = defaults.get("n_components")
            component_names = defaults.get("component_names")
            lower_bounds = defaults.get("lower_bounds")
            upper_bounds = defaults.get("upper_bounds")
            test_points = defaults.get("test_points")
            target_names = defaults.get("target_names")
            n_experiments = defaults.get("n_experiments")
            target_values = defaults.get("target_values")
            # fmt: off
            self.with_n_experiments(n=n_components)\
                .with_component_names(names=component_names)\
                .with_lower_bounds(bounds=lower_bounds)\
                .with_upper_bounds(bounds=upper_bounds)\
                .with_test_points(points=test_points)\
                .with_target_names(names=target_names)\
                .with_n_experiments(n=n_experiments)\
                .with_target_values(values=target_values)\
                .complete()
            # fmt: on
        elif defaults:
            self.complete()
        else:
            return

    @classmethod
    def build(cls: Type[ModelType]) -> ModelType:
        return cls()

    def with_n_components(self, n: Optional[int] = None) -> Self:
        if n is None:
            n = 3
        if n < 3:
            print(f"n_points must greater than 3, got {n}")
            return self
        self.n_components = n
        return self

    def with_component_names(
        self, names: Optional[ComponentNamesType] = None
    ) -> Self:
        if hasattr(self, "n_components"):
            if names is None:
                self.component_names = [
                    f"X{i}" for i in range(self.n_components)
                ]
            else:
                assert (
                    len(names) == self.n_components and len(names) >= 3
                ), "Number of components not match component names."
                self.component_names = names
        else:
            if names is None:
                # None of n_components and names provided. Choose defaults.
                print(
                    "Number of components and component names not found. Choose defaults automatically (3)."
                )
                self.with_n_components(3).with_component_names()
            else:
                assert (
                    len(names) >= 3
                ), "Number of components names must greater than 3."
                self.with_n_components(len(names)).with_component_names(names)
        return self

    def with_lower_bounds(self, bounds: Optional[BoundsType] = None) -> Self:
        if not hasattr(self, "n_components"):
            print("Number of components not set.")
            return self
        self.lower_bounds = self._check_bounds(
            bounds=bounds, n_components=self.n_components, type_="lower"
        )
        return self

    def with_upper_bounds(self, bounds: Optional[BoundsType] = None) -> Self:
        if not hasattr(self, "n_components"):
            print("Number of components not set.")
            return self
        self.upper_bounds = self._check_bounds(
            bounds=bounds, n_components=self.n_components, type_="upper"
        )
        return self

    def with_test_points(self, points: Optional[TestPointType] = None) -> Self:
        if points is None:
            points = 3
        # **TODO** Check test points.
        self.test_points = points
        return self

    def with_target_names(
        self, names: Optional[TargetNamesType] = None
    ) -> Self:
        if names is None:
            names = ["Target0"]
        self.target_names = names
        return self

    def with_n_experiments(self, n: Optional[int] = None) -> Self:
        if n is None:
            n = 1
        self.n_experiments = n
        return self

    def with_target_values(
        self, values: Optional[List[Tuple[Tuple[float]]]] = None
    ) -> Self:
        if values is None:
            self.target_values = None
        self.target_values = values
        return self

    def complete(self) -> Self:
        # If missing some params then make them default.
        if not hasattr(self, "n_components"):
            self.with_n_components()
        if not hasattr(self, "component_names"):
            self.with_component_names()
        if not hasattr(self, "lower_bounds"):
            self.with_lower_bounds()
        if not hasattr(self, "upper_bounds"):
            self.with_upper_bounds()
        if not hasattr(self, "test_points"):
            self.with_test_points()
        if not hasattr(self, "n_experiments"):
            self.with_n_experiments()
        if not hasattr(self, "target_names"):
            self.with_target_names()
        if not hasattr(self, "target_values"):
            self.with_target_values()

        self.experiment_points = self._generate_experiment_points()
        self.transform_matrix = self._generate_transform_matrix()
        # So called encoded proportion
        self.projected_matrix = self._generate_project_matrix()
        self.proportion = self._generate_proportion()
        # if test_points is not None:
        self.projected_matrix_test = self._generate_test_points(
            self.test_points
        )
        self.proportion_test = self.transform(self.projected_matrix_test)
        # So called real proportion
        self._response_surface_coefs: Optional[Dict[str, np.ndarray]] = None
        self.built = True
        return self

    @staticmethod
    def _check_bounds(
        bounds: Optional[BoundsType], n_components: int, type_: str
    ) -> np.ndarray[Fraction, Any]:
        make_fraction = lambda n: Fraction(str(n)).limit_denominator(
            LIMIT_DENOMINATOR
        )
        if bounds is None:
            bound = 0.0 if type_ == "lower" else 1.0
            bounds = tuple((make_fraction(bound),) * n_components)
        else:
            bounds = tuple(make_fraction(bound) for bound in bounds)
        return np.array(bounds)

    def _generate_proportion(self) -> PointsType:
        return self.transform(self.projected_matrix)

    def _generate_transform_matrix(self) -> np.ndarray[Fraction, Any]:
        # Get transform_matrix
        # Transform matrix
        lower_bounds = self.lower_bounds
        n = self.n_components
        # fmt: off
        transform_matrix = lower_bounds\
                .repeat(n)\
                .reshape((n, n))\
                + np.eye(n,dtype=Fraction)\
                * (1 - lower_bounds.sum())
        # fmt: on
        return transform_matrix

    def _generate_project_matrix(self) -> PointsType:
        """
        Take from experiment_points and divide component counts.
        """
        n = self.n_components
        experiment_points = self.experiment_points
        project_matrix = self.ndarray_to_fraction_array(
            np.ones((2**n - 1, n))
            * experiment_points
            / experiment_points.sum(axis=1)[:, np.newaxis]
        )

        return project_matrix

    def _generate_experiment_points(
        self,
    ) -> np.ndarray:
        """
        test_points:refer to proportion test_points.need to convert to nature points.
        """

        def bits(byte, n):
            """
            n-1 is total length of byte.
            for example, n=4, n-1=3, so,
            0b0001 -> 0001
            if use n, 0b0001 -> 00001
            """
            return [(byte & 1 << i) >> i for i in range(n - 1, -1, -1)]

        n = self.n_components
        # Generate a powerset mask except void set by start range from 1.
        experiment_points = np.array([bits(i, n=n) for i in range(1, 2**n)])
        return experiment_points

    def _generate_test_points(
        self, test_points: Optional[TestPointType] = 3
    ) -> PointsType:
        if isinstance(test_points, int):
            # random pick n_test_points does not need to transform_reverse
            added_points = np.random.dirichlet(
                (1,) * self.n_components, size=test_points
            )
        elif isinstance(test_points, List | Tuple):
            # if nested
            added_points = np.array(test_points)
            # if not nested
            if not all(isinstance(i, List | Tuple) for i in test_points):
                # expand one dim
                added_points = np.expand_dims(added_points, axis=0)
            added_points = self.transform_reverse(added_points)
        else:
            added_points = np.random.dirichlet((1,) * self.n_components, size=3)
        return self.ndarray_to_fraction_array(added_points)

    def transform(self, points: PointsType) -> PointsType:
        """
        Transform projected points to proportion.
        """
        return points @ self.transform_matrix.T

    def transform_reverse(self, points: PointsType) -> PointsType:
        """
        Transform proportion to projected space.
        """
        inv_matrix = np.linalg.inv(self.transform_matrix.T.astype(np.float64))
        ret = points @ inv_matrix
        return ret

    def transform_variables(self, projected: np.ndarray):
        """
        Transform variable(x) to producted variables.
        Theory:
            A: the coefficients;
            X: the variables;
            then we got
            y= a_1x_1 + a_2x_2 + ... a_nx_1x_2..x_n           (e.q-1)
            X as variables is known; y as response value is known. We want to solve A.
            We have to transform X to x_1,x_2,..,x_n,x_1x_2,..,x_1x_2..x_n form.
            Just take x_n from experiment points(masks), then make a product.
            We can get A by solve a linear equation.

        Parameters
        ----------
        projected: np.ndarray

        Returns
        -------
        Producted X matrix
        """
        # Numpy use bool array as mask
        points = self.experiment_points.astype(bool)
        transpose = projected.T
        return np.array([transpose[point].prod(axis=0) for point in points]).T

    @staticmethod
    def ndarray_to_fraction_array(arr: PointsType) -> PointsType:
        number_to_fraction = lambda x: Fraction(
            Decimal(str(x))
        ).limit_denominator(LIMIT_DENOMINATOR)
        return np.vectorize(number_to_fraction)(arr)

    def fit(self, y):
        """
        generate the formula with specific y, y be experiment's results
        assume y's order same as model.test_points
        @useage:
        model.fit(y)
        # coefficients of response surface
        # _response_surface_coef = []
        # for i, test_point in enumerate(self.test_points):
        #    r = len(test_point)
        #    temp = 0
        #    for j in range(1, r + 1):
        #        for test_point_pos in combinations(test_point, j):
        #            t = len(test_point_pos)
        #            # From 关颖男's 《混料试验设计》 Page:64
        #            temp += y[self.test_points.index(test_point_pos)] * \
        #                r * (-1)**(r - t) * t**(r - 1)
        #    _response_surface_coef.append(temp)
        # self._response_surface_coef = np.array(_response_surface_coef)
        """
        if len(y) != len(self.experiment_points):
            raise TypeError(
                "Missing required positional argument: "
                "y's length not match test_points"
            )
        # Response surface coefs
        X = self.transform_variables(self.projected_matrix).astype(np.float64)
        return np.linalg.solve(X, y)

    def fit_targets(self) -> Self:
        """
        Average target values by target names. Then fit them.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Average target values.
        values = np.array(self.target_values)
        values_len = values.shape[0]
        n = self.n_components
        target_names = self.target_names
        # Group mean.
        averaged = np.array(
            [values[i : i + n].T.mean(axis=1) for i in range(0, values_len, n)]
        )
        y = averaged[:, : len(self.experiment_points)]
        coefs = {name: self.fit(y[i]) for i, name in enumerate(target_names)}
        self._response_surface_coefs = coefs
        return self

    def predict(self, proportion: Iterable, target_name: str):
        if self._response_surface_coefs is None:
            print("Not fit yet.")
            return
        proportion = np.array(proportion)
        # Get coefficients of some target.
        coef = self._response_surface_coefs[target_name]
        # Convert proportion to projected space.
        projected_proportion = self.transform_reverse(proportion)
        prediction = self.transform_variables(projected_proportion) @ coef
        return prediction

    def predict_targets(self) -> Optional[Dict[str, np.ndarray]]:
        if self._response_surface_coefs is None:
            print("Not fit yet.")
            return
        proportion = np.vstack([self.proportion, self.proportion_test])
        predictions = dict()
        for target_name in self.target_names:
            predictions[target_name] = self.predict(proportion, target_name)
        return predictions

    # def score(self, X, y):
    #     return np.sum(np.abs(self.predict(X) - y)) / len(y)
    #
    # def __str__(self):
    #     if self._response_surface_coef is None:
    #         model_str = ""
    #     else:
    #         # ugly code NEED reform
    #         model_str = ("{:+.2f}*{}" * len(self.experiment_points)).format(
    #             *chain.from_iterable(
    #                 zip(
    #                     self._response_surface_coef,
    #                     [
    #                         ("z_{}*" * len(test_point)).format(*map(str, test_point))[
    #                             :-1
    #                         ]
    #                         for test_point in self.experiment_points
    #                     ],
    #                 )
    #             )
    #         )
    #     return model_str
    def save_to_file(self, file: str) -> None:
        workbook = openpyxl.Workbook()
        sheet_conditions = workbook.create_sheet(title="Conditions")
        sheet_experiments = workbook.create_sheet(title="Experiments")
        conditions_sheet_row_names = [
            "Model",
            "Names",
            "Lower bounds",
            "Upper_bounds",
            "Experiments numbers",
        ]
        for i, name in enumerate(conditions_sheet_row_names, start=1):
            sheet_conditions.cell(row=i, column=1, value=name)
        cell = sheet_conditions.cell(row=1, column=2, value=self.name)
        cell.number_format = openpyxl.styles.numbers.FORMAT_TEXT
        for i, point_name in enumerate(self.component_names, start=2):
            sheet_conditions.cell(row=2, column=i, value=point_name)
        for i, bound in enumerate(self.lower_bounds, start=2):
            cell = sheet_conditions.cell(row=3, column=i, value=str(bound))
            cell.number_format = openpyxl.styles.numbers.FORMAT_TEXT
        for i, bound in enumerate(self.upper_bounds, start=2):
            cell = sheet_conditions.cell(row=4, column=i, value=str(bound))
            cell.number_format = openpyxl.styles.numbers.FORMAT_TEXT
        cell = sheet_conditions.cell(row=5, column=2, value=self.n_experiments)
        cell.number_format = openpyxl.styles.numbers.FORMAT_NUMBER
        # proportion+proportion_test
        sheet_experiments.append(["№"] + self.component_names)
        experiment_ids = [
            "EXP-" + "".join(str(s) for s in p) for p in self.experiment_points
        ]
        experiment_ids = experiment_ids + [
            f"TEST-{i}"
            for i in range(
                self.proportion.shape[0] + self.proportion_test.shape[0]
            )
        ]
        for index, (experiment_id, line) in enumerate(
            zip(experiment_ids, chain(self.proportion, self.proportion_test)),
            start=1,
        ):
            row_index = index + 1
            # № row
            cell = sheet_experiments.cell(
                row=row_index, column=1, value=experiment_id
            )
            cell.number_format = openpyxl.styles.numbers.FORMAT_TEXT
            for line_index, number in enumerate(line, start=2):
                cell = sheet_experiments.cell(
                    row=row_index, column=line_index, value=float(number)
                )
                cell.number_format = openpyxl.styles.numbers.FORMAT_NUMBER_00
        target_col_start = 1 + self.n_components
        target_names_mix_with_n_experiments = []
        for target in self.target_names:
            for i in range(self.n_experiments):
                target_names_mix_with_n_experiments.append(target)
        for i, target in enumerate(
            target_names_mix_with_n_experiments, start=1
        ):
            col_index = target_col_start + i
            sheet_experiments.cell(row=1, column=col_index, value=target)
            if self.target_values is not None:
                for row_index, target_value in enumerate(
                    self.target_values[i - 1], start=2
                ):
                    sheet_experiments.cell(
                        row=row_index, column=col_index, value=target_value
                    )
        # Remove default sheet
        if "Sheet" in workbook.sheetnames:
            workbook.remove_sheet(workbook["Sheet"])
        try:
            workbook.save(file)
            workbook.close()
        except error as e:
            print(e)

    @classmethod
    def build_from_file(cls: Type[ModelType], file: str) -> Optional[ModelType]:
        """
        n_components: int | None = 3,
        component_names: Optional[ComponentNamesType] = None,
        lower_bounds: Optional[BoundsType] = None,
        upper_bounds: Optional[BoundsType] = None,
        test_points: Optional[TestPointType] = None,
        target_names: Optional[TargetNamesType] = None,
        experiment_num: Optional[int] = None,"""
        from openpyxl.cell.read_only import EmptyCell, ReadOnlyCell

        def get_cells_values(
            range: Iterable[ReadOnlyCell | EmptyCell],
        ) -> List[str]:
            return list(filter(lambda v: v, [cell.value for cell in range]))

        workbook = openpyxl.load_workbook(file, read_only=True, data_only=True)
        sheet_conditions = workbook["Conditions"]
        sheet_experiments = workbook["Experiments"]
        # Row 1 is for model name.
        component_names = get_cells_values(sheet_conditions[2][1:])
        n_components = len(component_names)
        lower_bounds = get_cells_values(sheet_conditions[3][1:])
        upper_bounds = get_cells_values(sheet_conditions[4][1:])
        n_experiments = get_cells_values(sheet_conditions[5][1:])
        if len(n_experiments) == 1:
            try:
                n_experiments = int(n_experiments[0])
            except:
                print("The Excel file is invalid.")
                print("Experiments numbers not right.")
                return
        else:
            print("The Excel file is invalid.")
            print("Experiments numbers not right.")
            return
        # Get target names and target test values.
        target_col_start = 1 + n_components
        target_names = []
        for target_cell in sheet_experiments[1][target_col_start:]:
            value = target_cell.value
            if value not in target_names:
                target_names.append(value)
        # Locate and load test points.
        row_As = []
        experiment_num = 0
        target_values = []
        for i, row in enumerate(sheet_experiments.iter_rows(), start=1):
            value = row[0].value
            if value == "№":
                continue
            if not value.startswith("EXP-") and not value.startswith("TEST-"):
                break
            if value.startswith("TEST-"):
                row_As.append(i)
            # Find target values.
            target_values.append(
                [float(cell.value) for cell in row[target_col_start:]]
            )
            experiment_num += 1
        # Transpose
        target_values = list(zip(*target_values))
        # Split into target names.
        # From Python documentation itertools -> grouper
        # target_values = list(
        #     zip(*([iter(target_values)] * n_experiments), strict=True)
        # )
        test_points = []
        for row_index in row_As:
            test_point = get_cells_values(
                sheet_experiments[row_index][1 : n_components + 1]
            )
            test_points.append(test_point)

        workbook.close()
        model = (
            cls.build()
            .with_n_components(n=n_components)
            .with_component_names(names=component_names)
            .with_lower_bounds(bounds=lower_bounds)
            .with_upper_bounds(bounds=upper_bounds)
            .with_test_points(points=test_points)
            .with_target_names(names=target_names)
            .with_n_experiments(n=n_experiments)
            .with_target_values(values=target_values)
            .complete()
        )
        return model

    def __repr__(self, param: Optional[str] = None):
        def get(obj: str) -> Any:
            if hasattr(self, obj):
                return getattr(self, obj)
            else:
                return "Not set yet."

        if not self.built:
            return "Model not built."
        if param:
            return f"{param}:\n{get(param)}"
        return f"""
ModelName:\n{get("name")}\n\n
NComponents:\n{get("n_components")}\n
Components:\n{get("component_names")}\n
LowerBounds:\n{get("lower_bounds")}\n
UpperBounds:\n{get("upper_bounds")}\n
Targets:\n{get("target_names")}\n
NExperiments:\n{get("n_experiments")}\n
TransformMatrix:\n{get("transform_matrix")}\n
ProjectedMatrix:\n{get("projected_matrix")}\n
ProjectedMatrixTest:\n{get("projected_matrix_test")}\n
Proportion:\n{get("proportion")}\n
ProportionTest:\n{get("proportion_test")}
"""

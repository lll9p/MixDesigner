#!/usr/bin/env python
# coding: utf-8
# import cmd
import gettext
import inspect
import locale
from typing import Any, Callable, Dict, List, Optional, Tuple

import openpyxl
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter, WordCompleter

import models
from models.types import ComponentNamesType, TargetNamesType, TestPointType
from utils import Completer

_ = gettext.gettext
try:
    loc = locale.getdefaultlocale()
    lang = loc[0]
    if (lang is not None) and (lang != "en_US"):
        l10n = gettext.translation(lang, localedir="locale", languages=[lang])
        l10n.install()
        _ = l10n.gettext
except Exception as e:
    print(e, "\nUsing defalt language - English(en_US.UFT-8)")


def parse(arg: str):
    _("Split args.")
    return tuple(arg.split())


def ignore(thing: Any) -> Any:
    return thing


def docstring(doc: str) -> Callable:
    def inner(obj):
        obj.__doc__ = doc
        return obj

    return inner


def info(*args: str) -> None:
    print(CLI.prompt_info, end="")
    print("".join(args))


def check_builder_point_names(
    prompt: Callable,
) -> Tuple[ComponentNamesType, int]:
    point_names = [f"X{i}" for i in range(3)]
    doc = _(
        "Tell me each component name. Or just tell me how many components.\n"
    )
    while (inputs := prompt(doc)) != "r":
        inputs = inputs.strip()
        if inputs == "":
            n_points = 3
            info(
                _(
                    "Input no name or number of components, I'd choose 3 for you."
                )
            )
            point_names = [f"X{i}" for i in range(n_points)]
            break
        try:
            # If no component names, just components' number.
            n_points = int(inputs)
            point_names = [f"X{i}" for i in range(n_points)]
            break
        except:
            point_names = inputs.split(" ")
            n_points = len(point_names)
            if n_points < 3:
                print(_("At least 3 components, please retry."))
                continue
        if len(set(point_names)) != n_points:
            info(_("Duplicates components'name, please retry."))
            continue
        else:
            return (point_names, n_points)
    return (point_names, 3)


def check_builder_bounds(
    prompt: Callable, n_points: int, lower: bool
) -> List[float]:
    lbdoc = (
        _("Type **lower** bounds, ") if lower else _("Type **upper** bounds, ")
    )
    doc = (
        lbdoc
        + _("like 'a b c' if you have 3 components.\n")
        + _(
            "If you leave it blank they would be all '0'(lower) or '1'(upper).\n"
        )
    )
    while (inputs := prompt(doc)) != "r":
        inputs = inputs.strip()
        if inputs == "":
            bounds = [0.0 if lower else 1.0] * n_points
            return bounds
        bounds = [float(x) for x in inputs.split(" ")]
        # need to check.
        if not all(map(lambda x: 0 <= x <= 1, bounds)):
            info(
                _(
                    "Input is out of bounds! Bounds must be in (0~1), please retry."
                )
            )
            continue
        if len(bounds) != n_points:
            info(_("Bounds must match components' shape, please retry."))
            continue
        else:
            return bounds
    return [0.0 if lower else 1.0] * n_points


def check_builder_test_points(prompt: Callable) -> TestPointType:
    doc = _(
        "Input test points, like 'a b c|d e f' or integer. Leave blank will choose 3 points automatically.\n"
    )
    while (inputs := prompt(doc)) != "r":
        inputs = inputs.strip()
        if not inputs:
            test_points = 3
        else:
            try:
                test_points = int(inputs)
            except:
                try:
                    test_points = [
                        [float(j) for j in i.split(" ")]
                        for i in inputs.split("|")
                    ]
                except:
                    info(_("Your input format not valid. Please retry."))
                    continue
        return test_points
    return 3


def check_builder_target_names(prompt: Callable) -> Optional[TargetNamesType]:
    doc = _(
        "Tell me each target name. Or just tell me how many targets.\n"
    ) + _("Leave it empty, I will choose 1 target by default.\n")
    while (inputs := prompt(doc)) != "r":
        inputs = inputs.strip()
        if inputs == "":
            targets = ["Target0"]
            info(
                _(
                    "Input no name or number of components, I'd choose 1 for you."
                )
            )
        try:
            # If no target names, just targets' number.
            target_nums = int(inputs)
            targets = [f"Target{i}" for i in range(target_nums)]
        except:
            targets = inputs.split(" ")
            target_nums = len(targets)
            if target_nums < 1:
                print(_("At least 1 target, please retry."))
                continue
        return targets
    return ["Target0"]


def check_builder_experiments_num(prompt: Callable) -> int:
    doc = _("Number of repeat experiment you wish.\n")
    while (inputs := prompt(doc)) != "r":
        inputs = inputs.strip()
        try:
            num = int(inputs)
        except:
            num = 1
        return num
    return 1


def func_params_len(func: Callable) -> int:
    return len(inspect.signature(func).parameters)


class CLI:
    intro: str = _(
        "Welcome to MixDesiner! To get help press '?' or type 'help'."
    )
    prompt: str = "> "
    prompt_info: str = ">>> "
    models = models.__models__
    # model_dict: Dict = {
    #     "SimplexCentroid": models.SimplexCentroid,
    #     "单形重心": models.SimplexCentroid,
    # }
    # model_name: str | None = None
    # model: models.SimplexCentroid | None = None

    def __init__(self) -> None:
        self.cmds = self.get_cmds()
        completer = self.build_completer()
        self.session = PromptSession(completer=completer)
        self.model = None

    def loop(self) -> None:
        print(self.intro)
        while True:
            try:
                line = self.session.prompt(self.prompt)
            except KeyboardInterrupt:
                # continue
                break
            except EOFError:
                break
            else:
                # Enter processing.
                self.process_cmd(line)

    def process_cmd(self, line: str):
        cmd, *args = line.strip(" ").split(" ")
        if cmd in self.cmds:
            func = self.cmds[cmd]
            func(args)
            # params_len = func_params_len(func)
            # if not args:
            #     # args is empty list
            #     func()
            # elif params_len == len(args):
            #     func(*args)
            # else:
            #     info(_("Command arguments not matched."))
        else:
            info(_("Invalid command: "), cmd)

    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def cmd_names(self) -> List[str]:
        return list(
            filter(lambda name: name.startswith("do_"), dir(self.__class__))
        )

    def get_cmds(self) -> Dict[str, Callable]:
        cmd_names = list(
            filter(lambda name: name.startswith("do_"), dir(self.__class__))
        )
        words = [name.lstrip("do_") for name in cmd_names]
        return {word: getattr(self, cmd) for word, cmd in zip(words, cmd_names)}

    def build_completer(self) -> Completer:
        data = dict()
        for name in self.cmds.keys():
            if name == "build":
                data[name] = WordCompleter(
                    words=list(self.models.keys()), ignore_case=True
                )
            elif name == ("save", "export", "load"):
                data[name] = PathCompleter(expanduser=True)
            else:
                data[name] = None
        completer = Completer.from_nested_dict(data)
        return completer

    @docstring(_("Get Help."))
    def do_help(self, args: List[str] | None = None):
        if args and len(args) == 1:
            cmd = args[0]
            if cmd not in self.cmds:
                info(_("No such command."), args[0])
                return
            print(f"  {cmd:<8}: {self.cmds[cmd].__doc__:<2}")
        else:
            print("All commands:")
            for cmd_name, cmd in self.cmds.items():
                print(f"  {cmd_name:<8}: {cmd.__doc__:<2}")

    @docstring(_("Build a model."))
    def do_build(self, args: List[str] | None = None):
        prompt = self.session.prompt
        if args and len(args) == 1:
            model_name = args[0]
        else:
            info(_("Your argument(s) not match command."), str(args))
            return
        if model_name not in self.models:
            info(
                _(
                    "You must choose model name in exist models. "
                    "Type 'ls' to show models."
                )
            )
            return
        component_names, n_components = check_builder_point_names(prompt)
        lower_bounds = check_builder_bounds(
            prompt, n_points=n_components, lower=True
        )
        upper_bounds = check_builder_bounds(
            prompt, n_points=n_components, lower=False
        )
        test_points = check_builder_test_points(prompt)
        target_names = check_builder_target_names(prompt)
        n_experiments = check_builder_experiments_num(prompt)
        info(_("Parameters are:"))
        info(_("Components names:"), str(component_names))
        info(_("Count of components:"), str(n_components))
        info(_("Lower bounds:"), str(lower_bounds))
        info(_("Upper bounds:"), str(upper_bounds))
        info(_("Test points:"), str(test_points))
        info(_("Target names:"), str(target_names))
        info(_("Experiments numbers:"), str(n_experiments))
        confirm = input(
            _("Press enter to confirm. Type 'r' to return/cancel.\n")
        )
        if confirm == "r":
            return
        self.model = self.models[model_name](
            dict(
                n_components=n_components,
                component_names=component_names,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                test_points=test_points,
                target_names=target_names,
                n_experiments=n_experiments,
            )
        )
        info(_("Model built successfully."))

    @docstring(_("Save model parameters to an excel workbook."))
    def do_save(self, args: List[str] | None = None):
        if args and len(args) == 1:
            file = args[0]
            if not file.endswith(".xlsx"):
                file += ".xlsx"
        else:
            info(_("You must provide a filename."))
            return
        if self.model is not None:
            self.model.save_to_file(file)
            info(_("File saved."))

    @docstring(_("Load model parameters from an excel workbook."))
    def do_load(self, args: List[str] | None = None):
        if args and len(args) == 1:
            file = args[0]
            if not file.endswith(".xlsx"):
                file += ".xlsx"
        else:
            info(_("You must provide a filename."))
            return
        workbook = openpyxl.load_workbook(file, read_only=True)
        sheet_conditions = workbook["Conditions"]
        model_name = sheet_conditions[1][1].value
        # Need Check valid.
        self.model = self.models[model_name].build_from_file(file)
        info(_("File loaded. Use print to check."))

    @docstring(_("Export current model to word document."))
    def do_export(self, args: List[str] | None = None):
        if args and len(args) == 1:
            file = args[0]
            if not file.endswith(".docx"):
                file += ".docx"
        else:
            info(_("You must provide a filename."))
            return
        pass

    @docstring(_("Analyze current model."))
    def do_analyze(self, args: List[str] | None = None):
        ignore(args)
        pass

    @docstring(_("Find optimal point."))
    def do_find(self, args: List[str] | None = None):
        ignore(args)
        if self.model is not None:
            pass
            # self.model.fit()

    @docstring(_("Show current model."))
    def do_print(self, params: List[str] | None = None):
        if self.model is not None:
            if (params is None) or params == [""]:
                print(self.model.__repr__())
            elif len(params) > 1:
                for arg in params:
                    print(self.model.__repr__(param=arg))
        else:
            info(_("Model not build. Please build a model."))

    @docstring(_("Show available models."))
    def do_ls(self, args: List[str] | None = None):
        ignore(args)
        info(" ".join(self.models.keys()))

    @docstring(_("Exit the program."))
    def do_quit(self, args: List[str] | None = None):
        ignore(args)
        info(_("Exiting MixDesiner, goodbye."))
        return exit(0)

    # Shortcuts
    do_q = do_quit

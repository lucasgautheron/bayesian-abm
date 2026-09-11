from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL_NAMES = (
    "add-model",
    "update-model",
    "add-summary-stat",
    "configure-summary-stats",
    "simulate",
    "inference",
    "model-comparison",
    "report",
    "test",
)


class CursorSkillTests(unittest.TestCase):
    def test_workshop_skills_are_explicitly_invoked(self) -> None:
        for name in SKILL_NAMES:
            with self.subTest(name=name):
                path = ROOT / ".cursor" / "skills" / name / "SKILL.md"
                text = path.read_text(encoding="utf-8")
                sections = text.split("---", maxsplit=2)
                self.assertEqual(sections[0], "")
                metadata = dict(
                    line.split(":", maxsplit=1)
                    for line in sections[1].strip().splitlines()
                )
                metadata = {
                    key.strip(): value.strip()
                    for key, value in metadata.items()
                }
                self.assertEqual(metadata["name"], name)
                self.assertTrue(metadata["description"])
                self.assertEqual(
                    metadata["disable-model-invocation"],
                    "true",
                )

    def test_pipeline_skills_use_the_registered_scripts(self) -> None:
        for name in ("simulate", "inference", "model-comparison"):
            with self.subTest(name=name):
                path = ROOT / ".cursor" / "skills" / name / "SKILL.md"
                text = path.read_text(encoding="utf-8")
                self.assertIn("scripts/aws/remote.py run --fallback-local", text)
                self.assertIn(f"python scripts/{name}.py <model>", text)
                self.assertIn("--cpus 16", text)
                self.assertIn("--cpus 4", text)
                self.assertIn("/configure-summary-stats", text)
                self.assertIn("models/__init__.py", text)
                self.assertNotIn("models.MODEL_REGISTRY", text)
                self.assertIn("Do not add `--show`", text)
                self.assertNotIn("visible terminal", text)
                self.assertIn("normal shell", text)
        comparison = (
            ROOT / ".cursor" / "skills" / "model-comparison" / "SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python scripts/model-comparison.py <model> <model>",
            comparison,
        )
        self.assertIn("two or more", comparison)

        report = (ROOT / ".cursor" / "skills" / "report" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("scripts/aws/remote.py run --fallback-local", report)
        self.assertIn(
            "scripts/aws/remote.py run -- python scripts/report.py <model>",
            report,
        )
        self.assertIn("--cpus 16", report)
        self.assertNotIn("--cpus 4", report)
        self.assertIn("/configure-summary-stats", report)
        self.assertIn("Do not add `--show`", report)
        self.assertNotIn("visible terminal", report)
        self.assertIn("Do not fall back locally", report)

    def test_local_test_skill_runs_the_ssh_probe(self) -> None:
        text = (
            ROOT / ".cursor" / "skills" / "test" / "SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("python scripts/test.py", text)
        self.assertIn("python scripts/test.py --verbose", text)
        self.assertIn("Do not start, stop, create, or destroy", text)
        self.assertIn("ssh-key", text)
        self.assertIn("instance", text)
        self.assertIn("connection", text)
        self.assertIn("in this order", text)
        self.assertIn("SSH", text)
        self.assertIn("remote.py check", text)
        self.assertNotIn("smoke test", text.lower())
        self.assertNotIn("simulation", text.lower())

    def test_report_skill_requires_markdown_and_reuses_training_data(
        self,
    ) -> None:
        text = (
            ROOT / ".cursor" / "skills" / "report" / "SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("report.md", text)
        self.assertIn("reuses the inference network's", text)
        self.assertIn("natural-scale mean, sigma", text)

    def test_configuration_recommends_defaults_without_an_options_menu(
        self,
    ) -> None:
        path = (
            ROOT
            / ".cursor"
            / "skills"
            / "configure-summary-stats"
            / "SKILL.md"
        )
        text = " ".join(path.read_text(encoding="utf-8").split())

        self.assertIn("explicitly recommend accepting that set", text)
        self.assertIn("Default to the `contacts` dataset", text)
        self.assertIn(
            "Ask which new summary statistic the participant would like "
            "to implement",
            text,
        )
        self.assertIn(
            "Do not proactively list alternative registered statistics",
            text,
        )
        self.assertIn(
            "only if the participant explicitly asks to use all of them",
            text,
        )

        add_model = " ".join(
            (
                ROOT / ".cursor" / "skills" / "add-model" / "SKILL.md"
            ).read_text(encoding="utf-8").split()
        )
        add_summary = " ".join(
            (
                ROOT / ".cursor" / "skills" / "add-summary-stat" / "SKILL.md"
            ).read_text(encoding="utf-8").split()
        )
        self.assertIn(
            "Assume the model belongs to `contacts` (first tutorial session)",
            add_model,
        )
        self.assertIn(
            "Assume the `contacts` dataset (first tutorial session)",
            add_summary,
        )

    def test_slash_skills_delegate_to_authoritative_workflows(self) -> None:
        model_workflow = (
            ROOT / "SKILLS" / "add-new-model" / "SKILL.md"
        ).read_text(encoding="utf-8")
        summary_workflow = (
            ROOT / "SKILLS" / "add-summary-statistic" / "SKILL.md"
        ).read_text(encoding="utf-8")
        add_model = (
            ROOT / ".cursor" / "skills" / "add-model" / "SKILL.md"
        ).read_text(encoding="utf-8")
        add_summary = (
            ROOT / ".cursor" / "skills" / "add-summary-stat" / "SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("SKILLS/add-new-model/SKILL.md", add_model)
        self.assertIn(
            "SKILLS/add-summary-statistic/SKILL.md",
            add_summary,
        )
        for requirement in (
            "Immediately switch to Plan mode",
            "`None` only when every free prior variable",
            "base.model.INTERVAL_SECONDS",
            "Do not add trivial accessor",
            "identical seeds produce identical parameters and contacts",
            "base.reporting import summarize_priors",
        ):
            with self.subTest(model_requirement=requirement):
                self.assertIn(requirement, model_workflow)
        for requirement in (
            "exactly one finite numeric scalar",
            "invariant to contact-row order",
            "empty intervals and isolated agents",
            "SUMMARY_BUILDERS",
        ):
            with self.subTest(summary_requirement=requirement):
                self.assertIn(requirement, summary_workflow)

    def test_superseded_always_applied_rule_is_removed(self) -> None:
        self.assertFalse(
            (
                ROOT
                / ".cursor"
                / "rules"
                / "summary-statistics.mdc"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()

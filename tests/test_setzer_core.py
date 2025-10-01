import unittest
from unittest import mock

import setzer_core
from setzer_core import Cue, _apply_batch


class FakeStreamResponse:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._lines)


class ApplyBatchTests(unittest.TestCase):
    def setUp(self):
        self.original_batch = [("1", "Hello"), ("2", "World"), ("3", "Foo"), ("4", "Bar")]
        self.cues = [
            Cue(index=1, start="0", end="1", text="Hello"),
            Cue(index=2, start="1", end="2", text="World"),
            Cue(index=3, start="2", end="3", text="Foo"),
            Cue(index=4, start="3", end="4", text="Bar"),
        ]

    def test_apply_batch_only_updates_present_ids(self):
        batches = iter([
            [("1", "Hola"), ("2", "Mundo")],
            [("3", "Baz"), ("4", "Qux")],
        ])

        def fake_batch(request_batch, **_):
            return next(batches)

        with mock.patch.object(setzer_core, "llm_translate_batch", side_effect=fake_batch):
            missing_first = _apply_batch(
                self.original_batch[:2],
                self.cues,
                "",
                "",
                "",
                "",
                "auto",
                False,
                30,
                True,
                None,
            )
            self.assertEqual(missing_first, [])
            self.assertEqual(self.cues[0].translated, "Hola")
            self.assertEqual(self.cues[1].translated, "Mundo")
            # Remaining cues untouched
            self.assertIsNone(self.cues[2].translated)
            self.assertIsNone(self.cues[3].translated)

            missing_second = _apply_batch(
                self.original_batch[2:],
                self.cues,
                "",
                "",
                "",
                "",
                "auto",
                False,
                30,
                True,
                None,
            )
            self.assertEqual(missing_second, [])
            self.assertEqual(self.cues[2].translated, "Baz")
            self.assertEqual(self.cues[3].translated, "Qux")

    def test_apply_batch_marks_missing_ids(self):
        def fake_batch(request_batch, **_):
            return [request_batch[0]]  # drop others

        with mock.patch.object(setzer_core, "llm_translate_batch", side_effect=fake_batch):
            missing = _apply_batch(
                self.original_batch[:2],
                self.cues,
                "",
                "",
                "",
                "",
                "auto",
                False,
                30,
                True,
                None,
            )
            self.assertEqual(missing, ["2"])
            self.assertEqual(self.cues[0].translated, "Hello")  # from fake translation
            self.assertEqual(self.cues[1].translated, "World")  # reused original text


    def test_apply_batch_treats_blank_translations_as_missing(self):
        def fake_batch(request_batch, **_):
            return [
                (request_batch[0][0], "  Hola  "),
                (request_batch[1][0], "   "),
            ]

        with mock.patch.object(setzer_core, "llm_translate_batch", side_effect=fake_batch):
            missing = _apply_batch(
                self.original_batch[:2],
                self.cues,
                "",
                "",
                "",
                "",
                "auto",
                False,
                30,
                True,
                None,
            )
            self.assertEqual(missing, ["2"])
            self.assertEqual(self.cues[0].translated, "  Hola  ")
            self.assertEqual(self.cues[1].translated, "World")


class HttpJsonStreamTests(unittest.TestCase):
    def test_streaming_data_prefix_handling(self):
        lines = [
            "data: {\"message\": {\"content\": \"Hel\"}}",
            "data: {\"message\": {\"content\": \"lo\"}}",
            "data: [DONE]",
        ]
        fake_response = FakeStreamResponse(lines)
        collector = []

        def fake_urlopen(req, timeout):
            return fake_response

        with mock.patch("setzer_core.urlopen", side_effect=fake_urlopen):
            result = setzer_core._http_json(
                "http://example/api/chat",
                {"a": 1},
                10,
                stream=True,
                raw_handler=collector.append,
            )

        self.assertEqual(result, "Hello")
        self.assertEqual(collector, lines)


class TranslationWhitespaceTests(unittest.TestCase):
    def test_llm_translate_single_preserves_whitespace(self):
        original = "  Hello world  \n"

        with mock.patch(
            "setzer_core._perform_llm_call",
            return_value="  Bonjour le monde  \n",
        ):
            translated = setzer_core.llm_translate_single(
                original,
                source="en",
                target="fr",
                model="gemma",
                server="http://example",
                translate_bracketed=True,
                llm_mode="chat",
                stream=False,
                timeout=10,
            )

        self.assertEqual(translated, "  Bonjour le monde  \n")

    def test_llm_translate_single_falls_back_on_blank_content(self):
        original = "Keep me"

        with mock.patch(
            "setzer_core._perform_llm_call",
            return_value="   \n\t  ",
        ):
            translated = setzer_core.llm_translate_single(
                original,
                source="en",
                target="fr",
                model="gemma",
                server="http://example",
                translate_bracketed=True,
                llm_mode="chat",
                stream=False,
                timeout=10,
            )

        self.assertEqual(translated, original)

    def test_llm_translate_batch_preserves_whitespace(self):
        pairs = [("1", "Hello"), ("2", "World")]
        response = "1|||  Salut  \n2|||Monde\n"

        with mock.patch(
            "setzer_core._perform_llm_call",
            return_value=response,
        ):
            translated_pairs = setzer_core.llm_translate_batch(
                pairs,
                source="en",
                target="fr",
                model="gemma",
                server="http://example",
                llm_mode="chat",
                stream=False,
                timeout=10,
                translate_bracketed=True,
            )

        self.assertEqual(translated_pairs, [("1", "  Salut  "), ("2", "Monde")])


if __name__ == "__main__":
    unittest.main()

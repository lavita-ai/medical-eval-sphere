import unittest


class TestLLMAnnotation(unittest.TestCase):

    def test_llm_annotation(self):
        import subprocess

        # Define the command and arguments as a list
        command = [
            "python", "../src/llm_annotation.py",
            "--input-path", "../data/assist/lavita_assist_processed_nov2024.csv",
            "--output-path", "../data/assist/lavita_assist_processed_v1_nov2024.csv",
            "--text-columns", "query",
            "--prompt-template", "../data/prompts/query_annotation_v1.txt",
            "--annotator-models", "openai_gpt-4o-2024-08-06",
            "--log-steps", "5"
        ]

        # Run the command
        subprocess.run(command, check=True)


if __name__ == '__main__':
    unittest.main()

import unittest


class TestLLMJudges(unittest.TestCase):

    def test_llm_judges(self):
        import subprocess

        # Define the command and arguments as a list
        command = [
            "python", "../src/llm_judge.py",
            "--llms", "openai_gpt-4o-2024-11-20", "anthropic_claude-3-5-sonnet-20241022"
        ]

        # Run the command
        subprocess.run(command, check=True)


if __name__ == '__main__':
    unittest.main()

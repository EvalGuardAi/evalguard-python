from setuptools import setup, find_packages

setup(
    name="evalguardai",
    version="2.0.1",
    description="Official EvalGuard Python SDK — LLM evaluation, red-team security, runtime guardrails, observability, and FinOps.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="EvalGuard",
    author_email="support@evalguard.ai",
    url="https://github.com/EvalGuardAi/evalguard",
    project_urls={
        "Homepage": "https://evalguard.ai",
        "Documentation": "https://docs.evalguard.ai/python-sdk",
        "Issues": "https://github.com/EvalGuardAi/evalguard/issues",
    },
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "langchain": ["langchain-core>=0.1.0"],
        "crewai": ["crewai>=0.1.0"],
        "bedrock": ["boto3>=1.28.0"],
        "fastapi": ["fastapi>=0.100.0"],
        "pydantic": ["pydantic>=2.5.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "langchain-core>=0.1.0",
            "crewai>=0.1.0",
            "boto3>=1.28.0",
            "fastapi>=0.100.0",
            "pydantic>=2.5.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-mock>=3.10",
            "responses>=0.23",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
        "Typing :: Typed",
    ],
    entry_points={
        "pytest11": ["evalguard = evalguard.pytest_plugin"],
        # R9-1: Pydantic plugin entry point — group is "pydantic" per
        # pydantic.plugin._loader.PYDANTIC_ENTRY_POINT_GROUP. Honors the
        # PYDANTIC_DISABLE_PLUGINS and EVALGUARD_PYDANTIC_DISABLED env
        # kill-switches.
        "pydantic": ["evalguard = evalguard.pydantic_integration:plugin"],
    },
    license="Apache-2.0",
    keywords=[
        "llm", "evaluation", "ai", "security", "red-team",
        "prompt-injection", "guardrails", "ai-safety", "evalguard",
        "openai", "anthropic", "langchain", "bedrock", "crewai",
    ],
)

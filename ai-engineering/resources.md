# AI Engineering Resources

Resources for building production AI systems and LLM applications.

## Books

- [Building LLMs for Production](https://www.oreilly.com/library/view/building-llms-for/9781098150952/) - Louis-François Bouchard, Louie Peters - Practical guide to deploying LLM applications
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann - Essential for understanding systems that support AI applications
- [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) - Jay Alammar, Maarten Grootendorst - Practical LLM application development

## Papers

- [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) - Anthropic - Comprehensive guide to prompt engineering
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) - Stanford NLP Group - Framework for optimizing LM programs
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - Packer et al. - Memory management for LLM agents
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al. - Foundational paper on agent reasoning patterns
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al. - Self-reflection for improving agent performance
- [The Rise and Potential of Large Language Model Based Agents](https://arxiv.org/abs/2309.07864) - Xi et al. - Comprehensive survey of LLM agents
- [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354) - Qin et al. - Survey on tool use in foundation models

## Courses

- [DeepLearning.AI - Building Systems with ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) - Andrew Ng - Building multi-step LLM workflows
- [DeepLearning.AI - LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) - Harrison Chase, Andrew Ng - Framework fundamentals
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) - The Full Stack - End-to-end LLM application development

## Articles

### Agent Systems
- [Anthropic's Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Anthropic - Design patterns for agent workflows
- [Building Autonomous Agents](https://www.deeplearning.ai/the-batch/tag/autonomous-agents/) - The Batch - Weekly coverage of agent developments
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph) - LangChain - Framework for building stateful multi-agent applications
- [OpenAI Agent Cookbook](https://cookbook.openai.com/examples/how_to_build_an_agent_with_the_node_sdk) - OpenAI - Practical agent implementation patterns

### Context Engineering
- [Anthropic's Long Context Window Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/long_context) - Anthropic - Techniques for working with 200K+ context windows
- [Context Engineering Strategies](https://www.pinecone.io/learn/context-engineering/) - Pinecone - Optimizing context for retrieval systems
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - DAIR.AI - Comprehensive guide to prompting techniques

### Evaluation-Driven Development
- [A Framework for LLM Evals](https://www.anthropic.com/research/measuring-model-persuasiveness) - Anthropic - Evaluation frameworks for LLM outputs
- [Beyond Accuracy: Evaluating LLM Applications](https://hamel.dev/blog/posts/evals/) - Hamel Husain - Practical evaluation strategies
- [Evaluating RAG Systems](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-2) - Anyscale - Metrics and testing for retrieval systems
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation) - LangChain - Tools and patterns for continuous evaluation
- [OpenAI Evals Framework](https://github.com/openai/evals) - OpenAI - Open-source framework for evaluating LLMs

### Memory Systems
- [Building Conversational Memory](https://www.pinecone.io/learn/conversational-memory/) - Pinecone - Implementing memory in chatbots
- [Long-term Memory for AI Agents](https://blog.langchain.dev/memory-for-llm-agents/) - LangChain - Patterns for persistent agent memory
- [MemGPT: Unlimited Context](https://memgpt.ai/) - Charles Packer - Virtual context management for LLMs
- [Memory in LangChain](https://python.langchain.com/docs/modules/memory/) - LangChain - Memory module documentation

### Model Context Protocol (MCP)
- [Anthropic MCP Documentation](https://modelcontextprotocol.io/) - Anthropic - Official protocol specification
- [Building MCP Servers](https://modelcontextprotocol.io/docs/tools/building) - Anthropic - Creating custom MCP integrations
- [MCP Quickstart](https://modelcontextprotocol.io/quickstart) - Anthropic - Getting started guide

### RAG (Retrieval Augmented Generation)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag/) - Pinecone - Beyond basic retrieval patterns
- [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) - LangChain - Video series on RAG fundamentals
- [The What and How of RAG](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) - DeepLearning.AI - RAG architecture overview

### Tool Use
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) - Anthropic - Claude-specific tool integration
- [Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling) - OpenAI - Design patterns for tool APIs
- [Gorilla: LLMs Connected with APIs](https://arxiv.org/abs/2305.15334) - Patil et al. - Fine-tuning LLMs for API calls

### DeepAgents & Multi-Agent Systems
- [AutoGPT Architecture](https://github.com/Significant-Gravitas/AutoGPT) - Significant Gravitas - Autonomous agent framework
- [CrewAI Documentation](https://docs.crewai.com/) - CrewAI - Multi-agent orchestration framework
- [MetaGPT: Multi-Agent Framework](https://arxiv.org/abs/2308.00352) - Hong et al. - Multi-agent programming framework
- [Multi-Agent Collaboration](https://arxiv.org/abs/2308.08155) - Li et al. - Patterns for agent communication
- [OpenAI Swarm](https://github.com/openai/swarm) - OpenAI - Lightweight multi-agent orchestration

## Frameworks & Tools

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Official Python SDK for Claude
- [DSPy](https://github.com/stanfordnlp/dspy) - Programming framework for LLM pipelines
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) - Library for building stateful multi-agent applications
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications
- [OpenAI Python SDK](https://github.com/openai/openai-python) - Official Python SDK for OpenAI APIs
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Microsoft's SDK for AI orchestration

## Community

- [LangChain Blog](https://blog.langchain.dev/) - Latest developments in LLM application patterns
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) - Open source LLM community
- [The Batch by DeepLearning.AI](https://www.deeplearning.ai/the-batch/) - Weekly AI newsletter covering applications

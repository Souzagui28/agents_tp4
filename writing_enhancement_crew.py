import os
from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool
from tools.user_text_retriever_tool import UserTextRetrieverTool

@CrewBase
class WritingEnhancementCrew:
    """Um crew para aprimorar a escrita do usuário com revisão, insights internos e externos."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        self.serper_tool = SerperDevTool()
        self.user_text_retriever_tool = UserTextRetrieverTool(
            text_folder_path='user_texts',
            persist_directory='chroma_db_user_texts'
        )
        self.llm = LLM(
            model=f"gemini/{os.environ['GEMINI_API_MODEL']}",
            api_key=os.environ["GOOGLE_API_KEY"]
        )

    @agent
    def agente_compilador_final(self) -> Agent:
        return Agent(
            config=self.agents_config['agente_compilador_final'],
            llm=self.llm,
            tools=[],
            allow_delegation=False,
            max_iter=5
        )

    @agent
    def agente_revisor_gramatical(self) -> Agent:
        return Agent(
            config=self.agents_config['agente_revisor_gramatical'],
            tools=[self.serper_tool, self.user_text_retriever_tool],
            llm=self.llm
        )

    @agent
    def agente_de_insights(self) -> Agent:
        return Agent(
            config=self.agents_config['agente_de_insights'],
            tools=[self.user_text_retriever_tool],
            llm=self.llm
        )

    @agent
    def agente_pesquisador(self) -> Agent:
        return Agent(
            config=self.agents_config['agente_pesquisador'],
            tools=[self.serper_tool],
            llm=self.llm
        )

    @task
    def master_text_enhancement_task(self) -> Task:
        return Task(config=self.tasks_config['master_text_enhancement_task'])

    @task
    def revision_task(self) -> Task:
        return Task(config=self.tasks_config['revision_task'])

    @task
    def internal_insights_task(self) -> Task:
        return Task(config=self.tasks_config['internal_insights_task'])

    @task
    def external_research_task(self) -> Task:
        return Task(config=self.tasks_config['external_research_task'])


    @crew
    def crew(self) -> Crew:
        tasks_list = self.tasks
        try:
            master_task = next(t for t in tasks_list if t.output_file == "output/relatorio_final_aprimoramento_texto.md")
            revision_task_obj = next(t for t in tasks_list if t.output_file == "output/revisao_gramatical.txt")
            insights_task_obj = next(t for t in tasks_list if t.output_file == "output/insights_internos.txt")
            research_task_obj = next(t for t in tasks_list if t.output_file == "output/sugestoes_externas.txt")
        except StopIteration:
            raise ValueError("Uma ou mais tarefas não foram encontradas...")
        
        # A ordem da lista é a ordem de execução
        ordered_tasks = [revision_task_obj, insights_task_obj, research_task_obj, master_task]

        return Crew(
            agents=self.agents,
            tasks=ordered_tasks,
            process=Process.sequential,
            verbose=True
        )
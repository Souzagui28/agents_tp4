# agents_tp4

Para executar a aplicação:

1. conda create --name tp4-souzagui python=3.10 -y
2. conda activate tp4-souzagui
3. pip install -r requirements.txt
4. copiar o .env.example e renomear para .env
5. colocar as chaves de API no .env

Chave de API Serper:
    https://serper.dev/signup
    https://serper.dev/api-keys

Chave de API Gemini Studio:
    https://aistudio.google.com/app/apikey


6. Executar as células que estão no arquivo notebook.ipynb
Obs: 
 - precisa selecionar o ambiente conda tp4-souzagui no vscode para poder funcionar (um ícone no canto superior direito do notebook jupyter permite alternar o ambiente).
 - pode acontecer de o kernel morrer na execução. Acredio que seja porque a API do Gemini às vezes não responde. Nesse caso, é preciso fechar o VS Code, abrir ele de novo e executar novamente as células do arquivo notebook.ipynb


Extra: 
 - conda deactivate para sair do ambiente
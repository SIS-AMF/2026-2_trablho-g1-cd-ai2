#!/usr/bin/env bash
# ==============================================================================
# COMANDOS PARA APRESENTAÇÃO — KALIMENTOS PDV
# Execute ou copie e cole os comandos abaixo um a um na hora da apresentação.
# ==============================================================================

# 0. Consultar catálogo de produtos e pagamentos válidos
python cli.py --listar

# 1. Cluster C0: Cuca Matinal (Abertura da Feira às 10h)
python cli.py --produto "Cuca Alemã" --hora 10 --preco 22.0 --quantidade 2 --metodo Pix --itens 1

# 2. Cluster C1: Doces Tradicionais (Tarde às 15h)
python cli.py --produto "Rapadura de Melado" --hora 15 --preco 7.0 --quantidade 3 --metodo Dinheiro --itens 1

# 3. Cluster C2: Rush das Cucas (Tarde às 16h)
python cli.py --produto "Cuca Enrolada" --hora 16 --preco 18.0 --quantidade 1 --metodo Cartão --itens 2

# 4. Cluster C3: Lanche Rápido / Alfajor (Tarde às 16h)
python cli.py --produto "Alfajor" --hora 16 --preco 6.0 --quantidade 1 --metodo Dinheiro --itens 1

# 5. Cluster C4: Atacado e Grandes Encomendas (Tarde às 14h)
python cli.py --produto "Rapadura Assada" --hora 14 --preco 10.0 --quantidade 80 --metodo Dinheiro --itens 5

# --- TESTES DE ERROS E GUARDRAILS ---

# 6. Teste de erro: Rodar sem parâmetros (Zero Defaults)
python cli.py

# 7. Teste de erro: Digitação com erro (Sugestão via difflib)
python cli.py --produto "alfajo" --hora 16 --preco 6.0 --quantidade 1 --metodo Dinheiro --itens 1

# 8. Teste de erro: Forma de pagamento inválida
python cli.py --produto "Alfajor" --hora 16 --preco 6.0 --quantidade 1 --metodo "Cheque" --itens 1

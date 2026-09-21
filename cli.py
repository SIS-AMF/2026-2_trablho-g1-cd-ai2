"""
CLI de Recomendação no Checkout — KAlimentos PDV
Disciplina: Inteligência Artificial II (Faculdade Antonio Meneghetti)

Uso:
    python cli.py --produto "Alfajor Preto" --hora 16 --preco 6.0 --quantidade 1 --metodo Dinheiro --itens 1
    python cli.py --produto "Cuca Alemã" --hora 10 --preco 22.0 --quantidade 2 --metodo Pix --itens 1
    python cli.py --listar
"""

import argparse
import difflib
import sys
import joblib
import pandas as pd

# 1. Carregar o modelo exportado pelo notebook eda_v2.ipynb
try:
    bundle = joblib.load("modelo_checkout.joblib")
except FileNotFoundError:
    print("Erro: 'modelo_checkout.joblib' não encontrado. Execute o notebook eda_v2.ipynb primeiro.", file=sys.stderr)
    sys.exit(1)

kmeans = bundle["kmeans"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
mapa_produtos = bundle["mapa_produtos"]
cluster_labels = bundle["cluster_labels"]
regras_recomendacao = bundle["regras_recomendacao"]
features_cluster = bundle["features"]

# Catálogo de produtos aceitos pelo PDV
produtos_validos = sorted(encoders["prod"].classes_)

# Métodos de pagamento suportados pelo modelo
metodos_validos = {
    "dinheiro": "Dinheiro",
    "cartao": "Cartão",
    "cartão": "Cartão",
    "pix": "Pix",
}


def listar_catalogo() -> None:
    """Exibe os produtos e formas de pagamento cadastrados no PDV."""
    print("=" * 60)
    print("PRODUTOS E FORMAS DE PAGAMENTO VÁLIDOS NO PDV")
    print("=" * 60)
    print("\nProdutos:")
    for p in produtos_validos:
        print(f"  • {p}")

    print("\nFormas de Pagamento:")
    for m in sorted(set(metodos_validos.values())):
        print(f"  • {m}")
    print("=" * 60)


def validar_produto(produto_str: str) -> str:
    """
    Valida se o produto existe no PDV. Se for válido, retorna o nome mapeado.
    Caso contrário, busca similares via difflib e encerra com mensagem de erro.
    """
    if not produto_str or not str(produto_str).strip():
        print("[ERRO] O parâmetro '--produto' não pode ser vazio.", file=sys.stderr)
        sys.exit(1)

    prod_limpo = str(produto_str).strip().replace(" ", "_").upper()
    prod_mapeado = mapa_produtos.get(prod_limpo, prod_limpo)

    if prod_mapeado in encoders["prod"].classes_:
        return prod_mapeado

    # Produto não reconhecido: buscar sugestões por aproximação
    similares = difflib.get_close_matches(prod_limpo, produtos_validos, n=3, cutoff=0.4)

    print("\n" + "!" * 65, file=sys.stderr)
    print(f"[ERRO] Produto '{produto_str}' não foi reconhecido no catálogo!", file=sys.stderr)
    if similares:
        sugestoes = ", ".join([f"'{s}'" for s in similares])
        print(f"👉 Você quis dizer: {sugestoes}?", file=sys.stderr)
    print("!" * 65, file=sys.stderr)

    print("\nProdutos aceitos:", file=sys.stderr)
    for p in produtos_validos:
        print(f"  • {p}", file=sys.stderr)
    print("\nDica: Utilize 'python cli.py --listar' para consultar o catálogo.\n", file=sys.stderr)
    sys.exit(1)


def validar_metodo_pagamento(metodo_str: str) -> str:
    """
    Valida se o método de pagamento é aceito e normaliza para a classe do modelo.
    """
    if not metodo_str or not str(metodo_str).strip():
        print("[ERRO] O parâmetro '--metodo' não pode ser vazio.", file=sys.stderr)
        sys.exit(1)

    chave = str(metodo_str).strip().lower()
    if chave in metodos_validos:
        return metodos_validos[chave]

    opcoes = ", ".join(sorted(set(metodos_validos.values())))
    print(f"\n[ERRO] Forma de pagamento '{metodo_str}' inválida!", file=sys.stderr)
    print(f"Opções permitidas: {opcoes}\n", file=sys.stderr)
    sys.exit(1)


def inferir(
    produto: str,
    hora: int,
    preco: float,
    quantidade: float,
    metodo: str,
    itens: int
):
    # 1. Validações estritas de domínio e tipos
    prod_mapeado = validar_produto(produto)
    metodo_normalizado = validar_metodo_pagamento(metodo)

    if hora is None or not (0 <= hora <= 23):
        print(f"[ERRO] Hora '{hora}' inválida. Informe um valor inteiro entre 0 e 23.", file=sys.stderr)
        sys.exit(1)

    if preco is None or preco <= 0:
        print(f"[ERRO] Preço 'R$ {preco}' inválido. O valor deve ser estritamente maior que zero.", file=sys.stderr)
        sys.exit(1)

    if quantidade is None or quantidade <= 0:
        print(f"[ERRO] Quantidade '{quantidade}' inválida. O valor deve ser estritamente maior que zero.", file=sys.stderr)
        sys.exit(1)

    if itens is None or itens < 1:
        print(f"[ERRO] Itens da cesta '{itens}' inválido. A diversidade da cesta deve ser no mínimo 1.", file=sys.stderr)
        sys.exit(1)

    # 2. Derivar o turno a partir da hora
    if hora <= 11:
        turno = "Manhã"
    elif hora <= 17:
        turno = "Tarde"
    else:
        turno = "Noite"

    # 3. Codificar variáveis categóricas com os encoders do treino
    le_prod = encoders["prod"]
    prod_code = le_prod.transform([prod_mapeado])[0]

    le_metodo = encoders["metodo"]
    metodo_code = le_metodo.transform([metodo_normalizado])[0]

    le_turno = encoders["turno"]
    turno_code = le_turno.transform([turno])[0]

    # 4. Montar vetor com as 8 features esperadas pelo modelo
    linha = pd.DataFrame([{
        "dia_horario": hora,
        "prod_code": prod_code,
        "metodo_code": metodo_code,
        "turno_code": turno_code,
        "preco_unitario_final": float(preco),
        "quantidade_final": float(quantidade),
        "valor_final_pedido": float(preco * quantidade),
        "qnt_repeticoes_num": float(itens),
    }])[features_cluster]

    # 5. Padronizar e predizer o cluster
    linha_scaled = scaler.transform(linha)
    cluster_id = int(kmeans.predict(linha_scaled)[0])

    return cluster_id, cluster_labels[cluster_id], regras_recomendacao[cluster_id], metodo_normalizado, turno


def main():
    parser = argparse.ArgumentParser(
        description="Recomendação de Produtos no Checkout do PDV — KAlimentos",
        add_help=True,
    )
    parser.add_argument("--produto", type=str, default=None, help="Nome do produto adicionado (obrigatório)")
    parser.add_argument("--hora", type=int, default=None, help="Hora inteira da venda entre 0 e 23 (obrigatório)")
    parser.add_argument("--preco", type=float, default=None, help="Preço unitário em R$ maior que zero (obrigatório)")
    parser.add_argument("--quantidade", type=float, default=None, help="Quantidade de itens maior que zero (obrigatório)")
    parser.add_argument("--metodo", type=str, default=None, help="Forma de pagamento: Dinheiro, Cartão, Pix (obrigatório)")
    parser.add_argument("--itens", type=int, default=None, help="Diversidade de itens na cesta do pedido >= 1 (obrigatório)")
    parser.add_argument("--listar", action="store_true", help="Exibir catálogo com todos os produtos e opções válidas")
    args = parser.parse_args()

    # Se solicitada apenas a consulta do catálogo
    if args.listar:
        listar_catalogo()
        sys.exit(0)

    # Verificação de presença obrigatória de todos os 6 parâmetros (SEM NENHUM DEFAULT)
    obrigatorios = {
        "--produto": args.produto,
        "--hora": args.hora,
        "--preco": args.preco,
        "--quantidade": args.quantidade,
        "--metodo": args.metodo,
        "--itens": args.itens,
    }

    ausentes = [flag for flag, val in obrigatorios.items() if val is None]
    if ausentes:
        print("\n" + "=" * 75, file=sys.stderr)
        print("[ERRO] Nenhum valor padrão é assumido. Parâmetro(s) obrigatório(s) ausente(s):", file=sys.stderr)
        for flag in ausentes:
            print(f"  • {flag}", file=sys.stderr)
        print("\nExemplo de uso correto:", file=sys.stderr)
        print('  python cli.py --produto "Alfajor Preto" --hora 16 --preco 6.0 --quantidade 1 --metodo Dinheiro --itens 1\n', file=sys.stderr)
        print("Dica: Use 'python cli.py --listar' para consultar os produtos aceitos.", file=sys.stderr)
        print("=" * 75 + "\n", file=sys.stderr)
        sys.exit(1)

    c_id, nome, rec, met_norm, turno = inferir(
        args.produto,
        args.hora,
        args.preco,
        args.quantidade,
        args.metodo,
        args.itens
    )

    total = args.preco * args.quantidade
    print("\n" + "=" * 70)
    print("           KALIMENTOS PDV — SUPORTE À DECISÃO NO CHECKOUT")
    print("=" * 70)
    print(f"Produto Registrado: {args.produto}")
    print(f"Horário e Turno   : {args.hora:02d}:00h ({turno})")
    print(f"Preço e Volume    : {args.quantidade:.0f} un x R$ {args.preco:.2f} = Total R$ {total:.2f}")
    print(f"Forma de Pagamento: {met_norm}")
    print(f"Cesta do Pedido   : {args.itens} item(ns) distintos na transação")
    print("-" * 70)
    print(f"Segmento Previsto : [{c_id}] {nome}")
    print(f"Ação Recomendada  : 👉 {rec}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

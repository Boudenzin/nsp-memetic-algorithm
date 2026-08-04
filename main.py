import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import seaborn as sns
from typing import List, Tuple
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# ==============================================================================
# 🧩 SEÇÃO 1: PESSOA 1 - CONFIGURAÇÃO DO PROBLEMA, PULP E FUNÇÃO DE FITNESS
# ==============================================================================

def carregar_parametros_problema():
    """Define os conjuntos, demandas e pesos de preferência do NSP."""
    avaliadores = [f"Prof_{i}" for i in range(1, 13)]  # 12 Avaliadores
    dias = [1, 2, 3, 4]                              # 4 Dias
    turnos = [1, 2, 3]                               # 1=Manhã, 2=Tarde, 3=Noite

    # Demanda r[t]: quantidade de professores necessários por turno (em cada dia)
    # Ex: 2 no turno da manhã (1), 2 à tarde (2) e 1 à noite (3)
    r = {1: 2, 2: 2, 3: 1}

    # Carga máxima de dias trabalhados por professor
    ca_valor = len(dias) - 1  # 3 dias no máximo (garante 1 folga)

    # Matriz de Preferências p[avaliador, turno, dia] (padrão peso 5)
    p = {(a, t, d): 5 for a in avaliadores for t in turnos for d in dias}

    # Preferências Específicas
    for d in dias:
        p["Prof_1", 1, d] = 9  # Prof_1 ama Manhã
        p["Prof_1", 3, d] = 1  # Prof_1 odeia Noite
        p["Prof_3", 3, d] = 9  # Prof_3 ama Noite

    for t in turnos:
        p["Prof_2", t, 1] = 0  # Prof_2 indisponível no Dia 1

    return avaliadores, dias, turnos, r, ca_valor, p


def resolver_modelo_pulp_referencia():
    """
    Executa o solver exato (PuLP/CBC) para obter a solução ótima matemática.
    Esta função serve como O'Ótimo Absoluto' para comparação.
    """
    avaliadores, dias, turnos, r, ca_valor, p = carregar_parametros_problema()
    prob = LpProblem("Escalonamento_ENIC_UFPB", LpMaximize)

    x = LpVariable.dicts("x", (avaliadores, turnos, dias), cat="Binary")

    # Função Objetivo: Maximizar Satisfação Total
    prob += lpSum(
        p[a, t, d] * x[a][t][d]
        for a in avaliadores
        for t in turnos
        for d in dias
    ), "Satisfacao_Total"

    # Restrição I: Cobertura
    for d in dias:
        for t in turnos:
            prob += lpSum(x[a][t][d] for a in avaliadores) == r[t]

    # Restrição II: Exclusividade Diária
    for a in avaliadores:
        for d in dias:
            prob += lpSum(x[a][t][d] for t in turnos) <= 1

    # Restrição III: Descanso Noturno (Zumbi)
    for a in avaliadores:
        for d in dias[:-1]:
            prob += x[a][3][d] + x[a][1][d + 1] <= 1

    # Restrição IV: Folga Obrigatória
    for a in avaliadores:
        prob += lpSum(x[a][t][d] for t in turnos for d in dias) <= ca_valor

    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(prob.objective)


def calcular_fitness(individuo, p, r, ca_valor):
    """
    Calcula a aptidão (fitness) de uma escala representada por uma matriz 2D.
    Matriz: Linhas = Avaliadores | Colunas = Dias | Valores = 0 (Folga), 1, 2, 3 (Turnos)
    """
    satisfacao = 0
    penalidades = 0
    PESO_PENALIDADE = 100  # Penalidade alta para desincentivar escalas inválidas

    n_avaliadores, n_dias = individuo.shape

    # 1. Avaliação da Satisfação (Função Objetivo)
    for a_idx in range(n_avaliadores):
        prof_nome = f"Prof_{a_idx+1}"
        for d_idx in range(n_dias):
            dia = d_idx + 1
            turno = individuo[a_idx, d_idx]
            if turno > 0:  # Se está escalado
                satisfacao += p.get((prof_nome, turno, dia), 5)

    # 2. Restrição I: Cobertura por Turno e Dia
    for d_idx in range(n_dias):
        for t in [1, 2, 3]:
            alocados = np.sum(individuo[:, d_idx] == t)
            demanda = r.get(t, 0)
            if alocados != demanda:
                penalidades += abs(alocados - demanda)

    # 3. Restrição III: Descanso Noturno (Noite no dia d -> Manhã no dia d+1)
    for a_idx in range(n_avaliadores):
        for d_idx in range(n_dias - 1):
            if individuo[a_idx, d_idx] == 3 and individuo[a_idx, d_idx + 1] == 1:
                penalidades += 1

    # 4. Restrição IV: Carga Máxima (Folga Obrigatória)
    for a_idx in range(n_avaliadores):
        dias_trabalhados = np.sum(individuo[a_idx, :] > 0)
        if dias_trabalhados > ca_valor:
            penalidades += (dias_trabalhados - ca_valor)

    return satisfacao - (penalidades * PESO_PENALIDADE)


# ==============================================================================
# 🧬 SEÇÃO 2: PESSOA 2 - NÚCLEO DO ALGORITMO GENÉTICO BASE (OPERADORES)
# ==============================================================================

def gerar_individuo(
    n_avaliadores: int,
    n_dias: int
) -> np.ndarray:
    """
    Gera um indivíduo aleatório.

    Cada posição da matriz representa o turno de um avaliador em um dia.

    0 = Folga
    1 = Manhã
    2 = Tarde
    3 = Noite
    """

    return np.random.randint(
        low=0,
        high=4,
        size=(n_avaliadores, n_dias)
    )

def gerar_populacao_inicial(
    tamanho_populacao: int,
    n_avaliadores: int,
    n_dias: int
) -> List[np.ndarray]:
    """
    Gera uma população inicial.
    """

    return [

        gerar_individuo(
            n_avaliadores,
            n_dias
        )

        for _ in range(tamanho_populacao)

    ]

def selecao_torneio(
    populacao: List[np.ndarray],
    p,
    r,
    ca_valor,
    tamanho_torneio: int = 3
) -> np.ndarray:
    """
    Seleciona o melhor indivíduo entre competidores sorteados.
    """

    competidores = random.sample(
        populacao,
        tamanho_torneio
    )

    return max(

        competidores,

        key=lambda individuo: calcular_fitness(
            individuo,
            p,
            r,
            ca_valor
        )

    ).copy()

def crossover_um_ponto(
    pai1: np.ndarray,
    pai2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza crossover de um ponto.
    """

    n_linhas = pai1.shape[0]

    ponto = random.randint(1, n_linhas - 1)

    filho1 = np.vstack((
        pai1[:ponto],
        pai2[ponto:]
    ))

    filho2 = np.vstack((
        pai2[:ponto],
        pai1[ponto:]
    ))

    return filho1, filho2


def crossover_dois_pontos(
    pai1: np.ndarray,
    pai2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza crossover de dois pontos.
    """

    n_linhas = pai1.shape[0]

    ponto1, ponto2 = sorted(
        random.sample(
            range(1, n_linhas),
            2
        )
    )

    filho1 = np.vstack((
        pai1[:ponto1],
        pai2[ponto1:ponto2],
        pai1[ponto2:]
    ))

    filho2 = np.vstack((
        pai2[:ponto1],
        pai1[ponto1:ponto2],
        pai2[ponto2:]
    ))

    return filho1, filho2


def realizar_crossover(
    pai1: np.ndarray,
    pai2: np.ndarray,
    taxa_crossover: float = 0.8,
    tipo: str = "2pontos"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Controla quando e qual crossover será utilizado.

    taxa_crossover:
        Probabilidade de ocorrer crossover.

    tipo:
        "1ponto" ou "2pontos"
    """

    # Não ocorre crossover
    if random.random() > taxa_crossover:
        return pai1.copy(), pai2.copy()

    # Escolhe o tipo
    if tipo == "1ponto":
        return crossover_um_ponto(
            pai1,
            pai2
        )

    return crossover_dois_pontos(
        pai1,
        pai2
    )

def mutacao(
    individuo: np.ndarray,
    taxa_mutacao: float = 0.05
) -> np.ndarray:
    """
    Aplica mutação gene a gene.

    Cada gene tem uma probabilidade de ser alterado.
    """

    filho = individuo.copy()

    n_avaliadores, n_dias = filho.shape

    for i in range(n_avaliadores):

        for j in range(n_dias):

            if random.random() <= taxa_mutacao:

                valores = np.array([0, 1, 2, 3])

                valores = valores[valores != filho[i, j]]

                filho[i, j] = np.random.choice(valores)

    return filho


def elitismo(
    populacao: List[np.ndarray],
    p,
    r,
    ca_valor,
    quantidade_elite: int = 2
) -> List[np.ndarray]:
    """
    Retorna os melhores indivíduos da população.
    """

    populacao_ordenada = sorted(
        populacao,
        key=lambda individuo: calcular_fitness(
            individuo,
            p,
            r,
            ca_valor
        ),
        reverse=True
    )

    return [
        individuo.copy()
        for individuo in populacao_ordenada[:quantidade_elite]
    ]


def gerar_nova_populacao(
    populacao: List[np.ndarray],
    p,
    r,
    ca_valor,
    taxa_mutacao: float = 0.05,
    taxa_crossover: float = 0.80,
    tipo_crossover: str = "2pontos",
    quantidade_elite: int = 2
) -> List[np.ndarray]:
    """
    Gera uma nova população utilizando:

    - Elitismo
    - Seleção por torneio
    - Crossover
    - Mutação
    """

    tamanho_populacao = len(populacao)

    nova_populacao = elitismo(
        populacao,
        p,
        r,
        ca_valor,
        quantidade_elite
    )

    while len(nova_populacao) < tamanho_populacao:

        # Seleção dos pais
        pai1 = selecao_torneio(
            populacao,
            p,
            r,
            ca_valor
        )

        pai2 = selecao_torneio(
            populacao,
            p,
            r,
            ca_valor
        )

        # Crossover
        filho1, filho2 = realizar_crossover(
            pai1,
            pai2,
            taxa_crossover,
            tipo_crossover
        )

        # Mutação
        filho1 = mutacao(
            filho1,
            taxa_mutacao
        )

        filho2 = mutacao(
            filho2,
            taxa_mutacao
        )

        nova_populacao.append(filho1)

        if len(nova_populacao) < tamanho_populacao:
            nova_populacao.append(filho2)

    return nova_populacao


# ==============================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# ==============================================================================

def algoritmo_genetico(
    n_avaliadores: int,
    n_dias: int,
    p,
    r,
    ca_valor,
    tamanho_populacao: int = 100,
    numero_geracoes: int = 200,
    taxa_mutacao: float = 0.05,
    taxa_crossover: float = 0.80,
    tipo_crossover: str = "2pontos",
    quantidade_elite: int = 2,
    paciencia: int = 30
):
    """
    Executa o Algoritmo Genético.

    Retorna:
        melhor_individuo
        melhor_fitness
        historico_melhor
        historico_media
    """

    # --------------------------------------------------------------------------
    # Geração da população inicial
    # --------------------------------------------------------------------------

    populacao = gerar_populacao_inicial(
        tamanho_populacao,
        n_avaliadores,
        n_dias
    )

    # --------------------------------------------------------------------------
    # Variáveis de controle
    # --------------------------------------------------------------------------

    melhor_individuo = None
    melhor_fitness = float("-inf")

    historico_melhor = []
    historico_media = []

    geracoes_sem_melhora = 0

    # --------------------------------------------------------------------------
    # Loop principal
    # --------------------------------------------------------------------------

    for geracao in range(numero_geracoes):

        # ----------------------------------------------------------
        # Calcula o fitness da população
        # ----------------------------------------------------------

        fitness_populacao = np.array([

            calcular_fitness(
                individuo,
                p,
                r,
                ca_valor
            )

            for individuo in populacao

        ])

        # ----------------------------------------------------------
        # Melhor indivíduo da geração
        # ----------------------------------------------------------

        indice_melhor = np.argmax(fitness_populacao)

        melhor_da_geracao = fitness_populacao[indice_melhor]

        media_da_geracao = np.mean(fitness_populacao)

        # ----------------------------------------------------------
        # Atualiza melhor solução encontrada
        # ----------------------------------------------------------

        if melhor_da_geracao > melhor_fitness:

            melhor_fitness = melhor_da_geracao

            melhor_individuo = populacao[indice_melhor].copy()

            geracoes_sem_melhora = 0

        else:

            geracoes_sem_melhora += 1

        # ----------------------------------------------------------
        # Salva histórico
        # ----------------------------------------------------------

        historico_melhor.append(melhor_fitness)

        historico_media.append(media_da_geracao)

        # ----------------------------------------------------------
        # Exibe progresso
        # ----------------------------------------------------------

        print(
            f"Geração {geracao + 1:03d}/{numero_geracoes}"
            f" | Melhor = {melhor_fitness:.2f}"
            f" | Média = {media_da_geracao:.2f}"
        )

        # ----------------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------------

        if geracoes_sem_melhora >= paciencia:

            print("\nEarly Stopping ativado.")
            print(f"Sem melhoria nas últimas {paciencia} gerações.")

            break

        # ----------------------------------------------------------
        # Nova geração
        # ----------------------------------------------------------

        populacao = gerar_nova_populacao(
            populacao,
            p,
            r,
            ca_valor,
            taxa_mutacao=taxa_mutacao,
            taxa_crossover=taxa_crossover,
            tipo_crossover=tipo_crossover,
            quantidade_elite=quantidade_elite
        )

    # --------------------------------------------------------------------------
    # Retorno
    # --------------------------------------------------------------------------

    return (
        melhor_individuo,
        melhor_fitness,
        historico_melhor,
        historico_media
    )


# ==============================================================================
# 🔍 SEÇÃO 3: PESSOA 3 - MÓDULO DE BUSCA LOCAL (MEMÉTICO / HILL CLIMBING)
# ==============================================================================

TURNOS_POSSIVEIS = [0, 1, 2, 3]  # 0 = Folga, 1 = Manhã, 2 = Tarde, 3 = Noite
 
 
def gerar_vizinho_troca_turno(individuo):
    """Move 1: muda o turno de UM professor em UM dia aleatório."""
    vizinho = individuo.copy()
    n_avaliadores, n_dias = vizinho.shape
    a = random.randrange(n_avaliadores)
    d = random.randrange(n_dias)
    turno_atual = vizinho[a, d]
    opcoes = [t for t in TURNOS_POSSIVEIS if t != turno_atual]
    vizinho[a, d] = random.choice(opcoes)
    return vizinho
 
 
def gerar_vizinho_swap_professores(individuo):
    """Move 2: troca os turnos entre DOIS professores no MESMO dia."""
    vizinho = individuo.copy()
    n_avaliadores, n_dias = vizinho.shape
    if n_avaliadores < 2:
        return vizinho
    d = random.randrange(n_dias)
    a1, a2 = random.sample(range(n_avaliadores), 2)
    vizinho[a1, d], vizinho[a2, d] = vizinho[a2, d], vizinho[a1, d]
    return vizinho
 
 
def gerar_vizinho_swap_dias(individuo):
    """Move 3: troca os turnos entre DOIS dias do MESMO professor.
    Bom para 'consertar' folga excedente ou turno zumbi movendo o problema
    de lugar sem violar a regra de 1 turno/dia."""
    vizinho = individuo.copy()
    n_avaliadores, n_dias = vizinho.shape
    if n_dias < 2:
        return vizinho
    a = random.randrange(n_avaliadores)
    d1, d2 = random.sample(range(n_dias), 2)
    vizinho[a, d1], vizinho[a, d2] = vizinho[a, d2], vizinho[a, d1]
    return vizinho
 
 
def gerar_vizinho(individuo):
    """Sorteia um dos 3 operadores de vizinhança e aplica."""
    movimento = random.choice([
        gerar_vizinho_troca_turno,
        gerar_vizinho_swap_professores,
        gerar_vizinho_swap_dias,
    ])
    return movimento(individuo)
 
 
def hill_climbing_first_improvement(individuo, p, r, ca_valor,
                                     max_iteracoes=200, max_vizinhos_por_iteracao=15):
    """
    Busca Local - estratégia FIRST-IMPROVEMENT.
    A cada iteração, gera vizinhos até achar um MELHOR que o atual e já aceita.
    Mais rápido, ótimo pra rodar em cima de muitos indivíduos do AG.
    """
    melhor = individuo.copy()
    melhor_fitness = calcular_fitness(melhor, p, r, ca_valor)
 
    for _ in range(max_iteracoes):
        melhorou = False
        for _ in range(max_vizinhos_por_iteracao):
            vizinho = gerar_vizinho(melhor)
            fit_vizinho = calcular_fitness(vizinho, p, r, ca_valor)
            if fit_vizinho > melhor_fitness:
                melhor = vizinho
                melhor_fitness = fit_vizinho
                melhorou = True
                break
        if not melhorou:
            break  # nenhum vizinho testado melhorou -> ótimo local, para
 
    return melhor, melhor_fitness
 
 
def hill_climbing_best_improvement(individuo, p, r, ca_valor,
                                    max_iteracoes=100, n_vizinhos=20):
    """
    Busca Local - estratégia BEST-IMPROVEMENT.
    A cada iteração, gera N vizinhos e escolhe o MELHOR entre eles.
    Mais lento, mas converge de forma mais "limpa" (bom pra comparar nos gráficos).
    """
    melhor = individuo.copy()
    melhor_fitness = calcular_fitness(melhor, p, r, ca_valor)
 
    for _ in range(max_iteracoes):
        candidatos = [gerar_vizinho(melhor) for _ in range(n_vizinhos)]
        fits = [calcular_fitness(c, p, r, ca_valor) for c in candidatos]
        idx_max = int(np.argmax(fits))
 
        if fits[idx_max] > melhor_fitness:
            melhor = candidatos[idx_max]
            melhor_fitness = fits[idx_max]
        else:
            break  # nenhum vizinho gerado melhorou -> ótimo local, para
 
    return melhor, melhor_fitness
 
 
def aplicar_busca_local_na_populacao(populacao, p, r, ca_valor,
                                      taxa_aplicacao=0.2, estrategia="first", **kwargs):
    """
    🔗 PONTE COM O AG (Pessoa 2) -- é isso que transforma o AG num Memético.
 
    populacao: lista de indivíduos (matrizes numpy), idealmente já ordenada
               do melhor pro pior fitness
    taxa_aplicacao: fração da população (a partir do topo/elite) que recebe
                     o refinamento local. Esse é o hiperparâmetro que a
                     Pessoa 4 vai variar (ex: 10%, 20%, 50%)
    estrategia: "first" ou "best"
    **kwargs: repassado pra função de hill climbing (max_iteracoes, etc.)
    """
    n_refinar = max(1, int(len(populacao) * taxa_aplicacao))
    nova_populacao = []
 
    funcao_hc = (hill_climbing_first_improvement if estrategia == "first"
                 else hill_climbing_best_improvement)
 
    for i, individuo in enumerate(populacao):
        if i < n_refinar:
            refinado, _ = funcao_hc(individuo, p, r, ca_valor, **kwargs)
            nova_populacao.append(refinado)
        else:
            nova_populacao.append(individuo)
 
    return nova_populacao


# ==============================================================================
# ⚙️ SEÇÃO 4: PESSOA 4 - ENGINE DE EXECUÇÃO, BENCHMARKS E EXPERIMENTOS
# ==============================================================================

def gerar_parametros_cenario(n_avaliadores: int, n_dias: int = 4, seed: int = None):
    
    """
    Gera dinamicamente o conjunto de parâmetros (avaliadores, dias, turnos,
    demanda r, carga máxima e matriz de preferências p) para QUALQUER tamanho
    de cenário. É essa função que permite criar os cenários Pequeno/Médio/
    Grande pedidos no trabalho, sem precisar reescrever tudo na mão como a
    Pessoa 1 fez para o cenário fixo de 12 professores.
    """
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
 
    avaliadores = [f"Prof_{i}" for i in range(1, n_avaliadores + 1)]
    dias = list(range(1, n_dias + 1))
    turnos = [1, 2, 3]
 
    # Demanda por turno escalada proporcionalmente ao tamanho do cenário
    # (mantém uma fração parecida de gente trabalhando por turno/dia
    # independente do cenário ser pequeno, médio ou grande)
    r = {
        1: max(1, round(n_avaliadores * 0.15)),  # manhã
        2: max(1, round(n_avaliadores * 0.15)),  # tarde
        3: max(1, round(n_avaliadores * 0.10)),  # noite
    }
 
    ca_valor = n_dias - 1  # garante ao menos 1 folga na semana
 
    # Preferências: peso-base 5 com ruído normal, simulando perfis
    # diferentes de avaliadores (uns preferem manhã, outros noite, etc.)
    p = {}
    for a in avaliadores:
        for t in turnos:
            for d in dias:
                p[(a, t, d)] = int(np.clip(np.random.normal(5, 2), 0, 9))
 
    return avaliadores, dias, turnos, r, ca_valor, p
 
 
def definir_cenarios_padrao(n_dias: int = 4):
    """
    Define os 3 cenários oficiais do experimento científico do artigo:
    Pequeno (10), Médio (20) e Grande (50 avaliadores).
    Usa seed fixa (42) para que os cenários sejam reprodutíveis entre
    todas as rodadas de teste.
    """
    return {
        "Pequeno (10 profs)": gerar_parametros_cenario(10, n_dias=n_dias, seed=42),
        "Médio (20 profs)": gerar_parametros_cenario(20, n_dias=n_dias, seed=42),
        "Grande (50 profs)": gerar_parametros_cenario(50, n_dias=n_dias, seed=42),
    }
 
 
def rodar_experimento(
    nome_cenario,
    avaliadores, dias, turnos, r, ca_valor, p,
    usar_memetico: bool = False,
    taxa_mutacao: float = 0.05,
    taxa_crossover: float = 0.80,
    tipo_crossover: str = "2pontos",
    tamanho_populacao: int = 100,
    numero_geracoes: int = 150,
    quantidade_elite: int = 2,
    paciencia: int = 30,
    taxa_aplicacao_busca_local: float = 0.2,
    estrategia_busca_local: str = "first",
    freq_busca_local: int = 10,   # a cada X gerações roda a busca local
    seed: int = None,
):
    """
    Executa UMA rodada completa do AG (puro ou memético, dependendo de
    usar_memetico) e devolve um dicionário com todas as métricas
    relevantes para comparação (fitness final, tempo, histórico de
    convergência etc.).
 
    Essa é a função "atômica" de experimento -- a bateria de testes
    (executar_bateria_experimentos) só fica chamando ela várias vezes
    variando os hiperparâmetros e os cenários.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
 
    n_avaliadores = len(avaliadores)
    n_dias = len(dias)
 
    populacao = gerar_populacao_inicial(tamanho_populacao, n_avaliadores, n_dias)
 
    melhor_individuo = None
    melhor_fitness = float("-inf")
    historico_melhor = []
    historico_media = []
    geracoes_sem_melhora = 0
 
    tempo_inicio = time.time()
    geracao = 0
 
    for geracao in range(numero_geracoes):
 
        fitness_populacao = np.array([
            calcular_fitness(ind, p, r, ca_valor) for ind in populacao
        ])
 
        indice_melhor = int(np.argmax(fitness_populacao))
        melhor_da_geracao = fitness_populacao[indice_melhor]
        media_da_geracao = float(np.mean(fitness_populacao))
 
        if melhor_da_geracao > melhor_fitness:
            melhor_fitness = melhor_da_geracao
            melhor_individuo = populacao[indice_melhor].copy()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1
 
        historico_melhor.append(melhor_fitness)
        historico_media.append(media_da_geracao)
 
        if geracoes_sem_melhora >= paciencia:
            break
 
        populacao = gerar_nova_populacao(
            populacao, p, r, ca_valor,
            taxa_mutacao=taxa_mutacao,
            taxa_crossover=taxa_crossover,
            tipo_crossover=tipo_crossover,
            quantidade_elite=quantidade_elite,
        )
 
        # --- Etapa Memética: aplica busca local periodicamente (não em
        # toda geração, pra não deixar o experimento lento demais) ---
        if usar_memetico and (geracao % freq_busca_local == 0):
            populacao_ordenada = sorted(
                populacao,
                key=lambda ind: calcular_fitness(ind, p, r, ca_valor),
                reverse=True,
            )
            populacao = aplicar_busca_local_na_populacao(
                populacao_ordenada, p, r, ca_valor,
                taxa_aplicacao=taxa_aplicacao_busca_local,
                estrategia=estrategia_busca_local,
            )
 
    tempo_total = time.time() - tempo_inicio
 
    return {
        "cenario": nome_cenario,
        "n_avaliadores": n_avaliadores,
        "memetico": usar_memetico,
        "taxa_mutacao": taxa_mutacao,
        "taxa_aplicacao_busca_local": taxa_aplicacao_busca_local if usar_memetico else np.nan,
        "estrategia_busca_local": estrategia_busca_local if usar_memetico else None,
        "melhor_fitness": melhor_fitness,
        "geracoes_executadas": geracao + 1,
        "tempo_execucao_s": tempo_total,
        "melhor_individuo": melhor_individuo,
        "historico_melhor": historico_melhor,
        "historico_media": historico_media,
    }
 
 
def executar_bateria_experimentos(
    cenarios: dict = None,
    taxas_mutacao=(0.05, 0.20),
    taxas_busca_local=(0.10, 0.50),
    n_repeticoes: int = 3,
    numero_geracoes: int = 150,
    tamanho_populacao: int = 100,
    verbose: bool = True,
):
    """
    🔬 Núcleo científico do trabalho (o que dá "peso de artigo" pra ele).
 
    Roda uma bateria de experimentos cruzando:
      - Cenários (Pequeno / Médio / Grande)
      - AG puro vs AG Memético (com busca local)
      - Taxas de mutação (ex: 5% vs 20%)
      - Taxas de aplicação de busca local (ex: 10% vs 50%) -- só p/ memético
      - Repetições (pra calcular média/desvio-padrão e dar robustez
        estatística aos resultados, já que o AG é estocástico)
 
    Retorna um pandas.DataFrame "cru" (uma linha por execução), pronto
    para ser resumido (resumir_resultados) e plotado pela Pessoa 5.
    """
    if cenarios is None:
        cenarios = definir_cenarios_padrao()
 
    resultados = []
 
    for nome_cenario, (avaliadores, dias, turnos, r, ca_valor, p) in cenarios.items():
 
        for taxa_mut in taxas_mutacao:
 
            # --- AG PURO (baseline, sem busca local) ---
            for rep in range(n_repeticoes):
                res = rodar_experimento(
                    nome_cenario, avaliadores, dias, turnos, r, ca_valor, p,
                    usar_memetico=False,
                    taxa_mutacao=taxa_mut,
                    numero_geracoes=numero_geracoes,
                    tamanho_populacao=tamanho_populacao,
                    seed=rep,
                )
                res["repeticao"] = rep
                resultados.append(res)
 
            # --- AG MEMÉTICO (varia também a taxa de busca local) ---
            for taxa_bl in taxas_busca_local:
                for rep in range(n_repeticoes):
                    res = rodar_experimento(
                        nome_cenario, avaliadores, dias, turnos, r, ca_valor, p,
                        usar_memetico=True,
                        taxa_mutacao=taxa_mut,
                        taxa_aplicacao_busca_local=taxa_bl,
                        numero_geracoes=numero_geracoes,
                        tamanho_populacao=tamanho_populacao,
                        seed=rep,
                    )
                    res["repeticao"] = rep
                    resultados.append(res)
 
            if verbose:
                print(f"[OK] Cenário '{nome_cenario}' | mutação={taxa_mut:.0%} concluído.")
 
    df = pd.DataFrame(resultados)
    return df
 
 
def resumir_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os resultados brutos (uma linha por repetição) em estatísticas
    (média, desvio-padrão, melhor caso, tempo médio) agrupadas por
    configuração. Essa tabela resumida é a que entra no artigo/slide final,
    e é o principal insumo pros gráficos comparativos da Pessoa 5.
    """
    colunas_grupo = [
        "cenario", "memetico", "taxa_mutacao", "taxa_aplicacao_busca_local"
    ]
 
    resumo = df.groupby(colunas_grupo, dropna=False).agg(
        fitness_medio=("melhor_fitness", "mean"),
        fitness_desvio=("melhor_fitness", "std"),
        fitness_maximo=("melhor_fitness", "max"),
        tempo_medio_s=("tempo_execucao_s", "mean"),
        geracoes_media=("geracoes_executadas", "mean"),
    ).reset_index()
 
    return resumo.sort_values(by="fitness_medio", ascending=False)

# ==============================================================================
# 📊 SEÇÃO 5: PESSOA 5 - VISUALIZAÇÃO DE DADOS, DASHBOARD E SLIDES
# ==============================================================================

def plotar_curvas_convergencia(historico_puro, historico_memetico):
    """
    Plota as curvas de convergência (Gerações vs Fitness) comparando
    o Algoritmo Genético Puro e o Algoritmo Memético.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(historico_puro, label="AG Puro", color="#e74c3c", linewidth=2.5)
    plt.plot(historico_memetico, label="AG Memético (com Busca Local)", 
             color="#2ecc71", linewidth=2.5, linestyle="--")
    
    plt.title("Curva de Convergência: AG Puro vs AG Memético", fontsize=16, fontweight='bold')
    plt.xlabel("Gerações", fontsize=12)
    plt.ylabel("Fitness (Qualidade da Escala)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plotar_heatmap_escala(individuo_final, avaliadores, dias):
    """
    Gera um gráfico de mapa de calor (heatmap) mostrando a tabela final da escala.
    Substitui os números da matriz (0, 1, 2, 3) pelos nomes dos turnos.
    """
    plt.figure(figsize=(12, 7))
    
    # Cores: 0=Folga (Cinza claro), 1=Manhã (Amarelo), 2=Tarde (Laranja), 3=Noite (Roxo)
    cmap = ListedColormap(['#ecf0f1', '#f1c40f', '#e67e22', '#9b59b6'])
    nomes_turnos = ['Folga', 'Manhã', 'Tarde', 'Noite']
    
    # Gera o heatmap base usando a matriz do indivíduo final
    ax = sns.heatmap(individuo_final, cmap=cmap, linewidths=1, linecolor='white',
                     annot=True, cbar=False, fmt="d",
                     xticklabels=[f"Dia {d}" for d in dias],
                     yticklabels=avaliadores)
    
    plt.title("Dashboard Final da Escala dos Professores", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Dias", fontsize=12, labelpad=10)
    plt.ylabel("Avaliadores", fontsize=12)
    
    # Mágica para trocar os números (0, 1, 2, 3) pelo texto no gráfico
    for t in ax.texts:
        valor = int(t.get_text())
        t.set_text(nomes_turnos[valor])
        t.set_fontsize(10)
        t.set_fontweight('bold')
        
        # Ajusta a cor da fonte para ficar legível dependendo do fundo
        if valor == 0:
            t.set_color('#7f8c8d')  # Texto escuro para "Folga"
        else:
            t.set_color('white')    # Texto branco para os outros turnos
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 🚀 EXECUÇÃO PRINCIPAL (MAIN) - LIGAÇÃO DE TODAS AS PARTES
# ==============================================================================

if __name__ == "__main__":
    print("Iniciando a resolução do Nurse Scheduling Problem (NSP)...")
    
    # 1. Carregar os parâmetros do cenário (Pessoa 1 / Pessoa 4)
    avaliadores, dias, turnos, r, ca_valor, p = carregar_parametros_problema()
    
    # 2. Executar o Algoritmo Genético PURO (Pessoa 2)
    print("\nExecutando Algoritmo Genético Puro...")
    resultado_puro = rodar_experimento(
        nome_cenario="Teste UFPB", avaliadores=avaliadores, dias=dias, 
        turnos=turnos, r=r, ca_valor=ca_valor, p=p,
        usar_memetico=False, numero_geracoes=80
    )
    
    # 3. Executar o Algoritmo MEMÉTICO com Busca Local (Pessoa 3)
    print("Executando Algoritmo Memético...")
    resultado_memetico = rodar_experimento(
        nome_cenario="Teste UFPB", avaliadores=avaliadores, dias=dias, 
        turnos=turnos, r=r, ca_valor=ca_valor, p=p,
        usar_memetico=True, numero_geracoes=80
    )
    
    # 4. Gerar Gráficos e Dashboard (Pessoa 5)
    print("\nGerando Dashboard Visual...")
    
    # Plota o comparativo
    plotar_curvas_convergencia(
        resultado_puro["historico_melhor"], 
        resultado_memetico["historico_melhor"]
    )
    
    # Plota a escala do melhor indivíduo encontrado pelo Memético
    melhor_escala = resultado_memetico["melhor_individuo"]
    plotar_heatmap_escala(melhor_escala, avaliadores, dias)

# ==============================================================================
# 🚀 EXECUÇÃO PRINCIPAL (MAIN)
# ==============================================================================


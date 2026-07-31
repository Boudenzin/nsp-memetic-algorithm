import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import seaborn as sns

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



# ==============================================================================
# 📊 SEÇÃO 5: PESSOA 5 - VISUALIZAÇÃO DE DADOS, DASHBOARD E SLIDES
# ==============================================================================



# ==============================================================================
# 🚀 EXECUÇÃO PRINCIPAL (MAIN)
# ==============================================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstração do VE Map Optimizer com foco em confiabilidade
Testa o sistema que modifica apenas células com dados suficientes
"""

import time
import pandas as pd
import numpy as np
from ve_map_optimizer import VEMapOptimizerReliable

def main():
    print("🎯 VE MAP OPTIMIZER - OTIMIZAÇÃO CONFIÁVEL")
    print("=" * 60)
    print("📋 Sistema que modifica apenas células com dados suficientes")
    print("   ✅ Usa apenas condições estáveis (TPS estável, sem DFCO)")
    print("   ✅ Considera delay da sonda lambda")
    print("   ✅ Requer mínimo de pontos por célula")
    print("   ✅ Aplica fator conservador nas correções")
    print()
    
    start_time = time.time()
    
    # Inicializa otimizador
    optimizer = VEMapOptimizerReliable()
    optimizer.min_points_per_cell = 12  # Mínimo de pontos para confiabilidade
    
    # Tenta arquivo longo primeiro, depois curto
    test_files = [
        ("logs/long.msl", "Arquivo LONGO"),
        ("logs/short.msl", "Arquivo CURTO")
    ]
    
    msl_file = None
    for file_path, description in test_files:
        try:
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"📂 {description} encontrado: {file_path} ({lines} linhas)")
            msl_file = file_path
            break
        except FileNotFoundError:
            print(f"⚠️  {description} não encontrado: {file_path}")
    
    if not msl_file:
        print("❌ Nenhum arquivo MSL encontrado!")
        return
    
    # Carrega dados
    print(f"\n📊 CARREGANDO DADOS...")
    load_start = time.time()
    
    success = optimizer.load_data(
        msl_file=msl_file,
        ve_table_file="logs/veTable1Tbl_pre.table",
        lambda_table_file="logs/lambdaTable1Tbl_.table"
    )
    
    if not success:
        print("❌ Falha no carregamento dos dados!")
        return
    
    load_time = time.time() - load_start
    print(f"⏱️  Carregamento concluído em {load_time:.1f}s")
    
    # Estatísticas iniciais
    print(f"\n📈 ESTATÍSTICAS DOS DADOS:")
    stats = optimizer.processor.get_summary_stats()
    print(f"   Total de registros: {stats['total_points']:,}")
    print(f"   Registros estáveis: {stats.get('stable_points', 0):,} ({100*stats.get('stability_ratio', 0):.1f}%)")
    print(f"   Células com dados suficientes: {stats.get('covered_cells', 0)}/{stats.get('total_cells', 0)} ({100*stats.get('coverage_ratio', 0):.1f}%)")
    print(f"   Delay da sonda estimado: {optimizer.lambda_delay:.3f}s")
    
    # Gera heatmap de cobertura
    print(f"\n📊 Gerando heatmap de cobertura...")
    optimizer.processor.plot_coverage_heatmap("coverage.png")
    
    # Prepara dados de treinamento
    print(f"\n🔄 PREPARANDO DADOS DE TREINAMENTO...")
    prep_start = time.time()
    
    training_data = optimizer.prepare_training_data()
    
    if training_data is None:
        print("❌ Falha na preparação dos dados de treinamento!")
        return
    
    prep_time = time.time() - prep_start
    print(f"⏱️  Preparação concluída em {prep_time:.1f}s")
    
    # Treina modelo
    print(f"\n🤖 TREINANDO MODELO DE MACHINE LEARNING...")
    train_start = time.time()
    
    if not optimizer.train_model(training_data):
        print("❌ Falha no treinamento do modelo!")
        return
    
    train_time = time.time() - train_start
    print(f"⏱️  Treinamento concluído em {train_time:.1f}s")
    
    # Otimiza tabela VE
    print(f"\n🎯 OTIMIZANDO TABELA VE...")
    opt_start = time.time()
    
    # Usa fator conservador de 60%
    if not optimizer.optimize_ve_table(conservative_factor=0.6):
        print("❌ Falha na otimização da tabela!")
        return
    
    opt_time = time.time() - opt_start
    print(f"⏱️  Otimização concluída em {opt_time:.1f}s")
    
    # Salva resultados
    print(f"\n💾 SALVANDO RESULTADOS...")
    
    # Nome base do arquivo
    base_name = "long" if "long.msl" in msl_file else "short"
    
    # Salva tabela otimizada
    table_file = f"ve_optimized.table"
    if optimizer.save_optimized_table(table_file, "logs/veTable1Tbl_pre.table"):
        print(f"   ✅ Tabela VE: {table_file}")
    
    # Gera análise visual
    analysis_file = f"ve_analysis.png"
    optimizer.plot_optimization_analysis(analysis_file)
    print(f"   ✅ Análise visual: {analysis_file}")
    
    # Salva dados de treinamento para análise
    training_file = f"training_data.csv"
    training_data.to_csv(training_file, index=False)
    print(f"   ✅ Dados de treinamento: {training_file}")
    
    # Resumo final
    total_time = time.time() - start_time
    
    print(f"\n🏁 RESUMO FINAL:")
    print(f"   ⏱️  Tempo total: {total_time:.1f}s")
    print(f"   📊 Pontos processados: {stats['total_points']:,}")
    print(f"   🎯 Pontos estáveis: {stats.get('stable_points', 0):,}")
    print(f"   📈 Pontos ML: {len(training_data):,}")
    print(f"   🔧 Células modificadas: {optimizer.modification_stats['modified_cells']}")
    print(f"   ⚠️  Mudanças significativas: {optimizer.modification_stats['significant_changes']}")
    print(f"   📏 Maior mudança: {optimizer.modification_stats['max_change']:.1f}%")
    print(f"   🎯 Fator conservador: 60%")
    
    print(f"\n✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"   📋 Apenas células com dados suficientes foram modificadas")
    print(f"   🔍 Verifique os arquivos gerados para análise detalhada")

if __name__ == "__main__":
    main()

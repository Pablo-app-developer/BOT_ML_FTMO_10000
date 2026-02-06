# 🏆 Antigravity FTMO Bot - ML Trading System

**Estado:** ✅ Sistema V3 con Imitation Learning completado

## 🎯 Objetivo

Pasar el desafío de fondeo FTMO ($10,000) utilizando Machine Learning y gestión de riesgo estricta basada en Imitation Learning.

### Reglas FTMO
- **Profit Target:** $1,000 (10%)
- **Max Daily Loss:** $500 (5%)
- **Max Total Loss:** $1,000 (10%)

## 📊 Evolución del Proyecto

### ❌ V1 - Primer Intento (Overfitting Severo)
- **Entrenamiento:** 2M pasos en datos de BTC únicamente
- **Resultado Test Inicial:** 100% éxito (20/20)
- **Resultado Validación Real:** 0% éxito (0/30) - Overfitting total
- **Problema:** Memorizó datos de entrenamiento

### ❌ V2 - Anti-Overfitting
- **Mejoras:** Multi-asset data, early stopping, red más pequeña
- **Resultado:** 130k pasos, early stop
- **Problema:** Aprendió a "no hacer nada" (solo SELL en vacío)

### ✅ V3 - Imitation Learning (ACTUAL)
- **Enfoque:** Behavioral Cloning + RL
- **Datos:** 28,203 velas mezcladas (BTC, ETH, SOL)
- **Estrategia Experta:** SMA crossover con gestión FTMO
- **Resultado:** 300k pasos completados

## 🏗️ Arquitectura

```
antigravity-ftmo-10k/
├── src/
│   ├── strategy/
│   │   ├── ftmo_env.py         # Environment V1
│   │   ├── ftmo_env_v2.py      # Environment V2 (mejor reward)
│   │   ├── expert_strategy.py  # Estrategia experta SMA
│   │   ├── train.py            # Training básico
│   │   └── train_ftmo.py       # Training V1
│   └── utils/
│       └── download_data.py    # Descarga datos de Binance vía CCXT
├── train_v2.py                 # Training V2 (multi-asset)
├── train_imitation.py          # Training V3 (imitation learning)
├── evaluate_quick.py           # Evaluación rápida de modelos
├── validate_rigorous.py        # Validación rigurosa multi-asset
├── download_validation_data.py # Descarga datos frescos
└── requirements.txt
```

## 🚀 Quickstart

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Descargar Datos

```bash
# Datos de entrenamiento (6 meses BTC)
python src/utils/download_data.py

# Datos de validación (60 días BTC, ETH, SOL)
python download_validation_data.py
```

### 3. Entrenar Modelo

```bash
# Sistema V3 - Imitation Learning (Recomendado)
python train_imitation.py
```

### 4. Evaluar

```bash
# Evaluación rápida
python evaluate_quick.py

# Validación rigurosa multi-asset
python validate_rigorous.py
```

## 📈 Resultados

### V1 - Datos Training
- ✅ **Tasa de éxito:** 100% (20/20)
- ✅ **Ganancia promedio:** +10.21%

### V1 - Datos Frescos (Validación Real)
- ❌ **BTC (60d):** 0% (pérdida -2.05%)
- ❌ **ETH (60d):** 0% (pérdida -5.69%)
- ❌ **SOL (60d):** 0% (pérdida -10.19%)
- ❌ **Conclusión:** Overfitting severo

### V3 - Imitation Learning
- ✅ **Entrenamiento completado:** 301k pasos
- 📊 **Demonstraciones generadas:** ~40k ejemplos de estrategia experta
- ⏱️ **Tiempo:** ~10 minutos
- 🔄 **Próximo paso:** Validación pendiente

## 🛠️ Tecnologías

- **Python:** 3.10+
- **RL Framework:** Stable-Baselines3 (PPO)
- **Data Source:** CCXT (Binance)
- **Environment:** Gymnasium (OpenAI Gym)
- **Deep Learning:** PyTorch
- **Visualization:** TensorBoard

## 📚 Componentes Clave

### FTMOTradingEnvV2
Environment de Gymnasium que simula trading con reglas FTMO integradas:
- Penaliza violaciones de límites
- Premia gestión de riesgo
- Integra indicadores técnicos

### Expert Strategy
Estrategia conservadora basada en:
- SMA 20/50 crossover
- Filtro RSI (< 70 para compra)
- Stop Loss: 2%
- Take Profit: 4% (2:1 R:R)
- Position sizing: 30% máx

### Imitation Learning Pipeline
1. **Generar Demos:** Estrategia experta corre en datos históricos
2. **Entrenar RL:** Bot aprende de demos + mejora con RL
3. **Validar:** Test en datos completamente unseen

## ⚠️ Lecciones Aprendidas

1. **Más datos ≠ Mejor:** 2M pasos en datos limitados = overfitting
2. **Diversidad > Cantidad:** Multi-asset mezclado > Single asset largo
3. **Reward Shaping:** Premiar proceso > premiar resultado exacto
4. **Validación Rigurosa:** SIEMPRE validar en datos frescos unseen

## 🔜 Próximos Pasos

1. ✅ Validar V3 en datos frescos
2. 📊 Paper trading en testnet
3. 🎯 Si pasa validación → FTMO Challenge real
4. 🔄 Refinamiento continuo con datos nuevos

## 📖 Documentación Adicional

Ver archivos en el repositorio:
- `ARCHITECTURE_V2.md` - Detalles técnicos
- `HISTORIAL_DE_FASES.md` - Evolución del proyecto

## 🤝 Contribuciones

Proyecto personal para aprendizaje de ML en trading. 

## ⚖️ Disclaimer

Este bot es experimental. Trading con riesgo. No usar dinero real sin validación extensiva.

---

**Última actualización:** 2026-02-05  
**Estado:** Sistema V3 completado, validación pendiente

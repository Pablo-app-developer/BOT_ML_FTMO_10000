# MANUAL DE OPERACIONES — FTMO Challenge Bot

## QUE HACE ESTE BOT

Opera 3 pares de forma automatica en tu cuenta de MetaTrader 5:

| Celda | Simbolo | Estrategia | Riesgo |
|-------|---------|-----------|--------|
| 1 | XAUUSD (Oro) | Tendencia H1 con ATR | 1.25% |
| 2 | USDJPY | Tendencia H1 con ATR | 1.25% |
| 3 | EURUSD | Bollinger Band Squeeze | 1.25% |

Riesgo total maximo simultaneo: **3.75%** (si las 3 celdas abren a la vez).

---

## REQUISITOS ANTES DE ARRANCAR

1. **MetaTrader 5 abierto** e iniciado sesion en tu cuenta FTMO
2. **Python instalado** con los paquetes: `MetaTrader5`, `pandas`, `numpy`, `requests`
3. Conexion a internet activa

---

## COMO ARRANCAR

**Opcion 1 — Doble click:**
Ejecuta el archivo `ARRANCAR_BOT.bat` en la carpeta del proyecto.

**Opcion 2 — PowerShell:**
```
cd C:\Users\Personal\Documents\antigravity-ftmo-10k
python live_bot.py
```

Al arrancar veras en pantalla:
```
MT5 connected. Symbols: XAUUSD, USDJPY, EURUSD
Account #XXXXXXXX  Balance=$10025.53  Server=MetaQuotes-Demo
Bot started | initial=$10000.00 | risk=1.25%/cell ...
```
Si ves esas lineas, **el bot esta funcionando**.

---

## COMO PARAR

- **Normal:** Cierra la ventana de PowerShell o presiona `Ctrl+C` — el bot cierra todas las posiciones abiertas antes de salir.
- **Emergencia:** Cierra manualmente las posiciones desde MT5 directamente.

---

## HORARIO DE OPERACION

| Evento | Hora (broker) |
|--------|--------------|
| Inicio de sesion | 07:00 |
| Cierre de posiciones EOD | 20:55 |
| Fin de sesion | 21:00 |

Fuera de ese horario el bot espera sin operar. Esto es normal.

---

## PROTECCIONES FTMO (AUTOMATICAS)

El bot para solo si se activa alguna de estas:

| Proteccion | Limite interno | Limite FTMO real | Que hace |
|-----------|---------------|-----------------|---------|
| Perdida diaria | -2.5% del dia | -5% | Cierra todo, reanuda manana |
| Perdida total | -8% del initial | -10% | Cierra todo, **para para siempre** |
| Objetivo alcanzado | +11% con 4+ dias | +10% | Cierra todo, **para para siempre** |

Cuando para "para siempre" significa que alcanzo el objetivo o proteccion total — **no lo reinicies sin revisar el log primero**.

---

## COMO VER EL LOG EN TIEMPO REAL

Abre otra ventana de PowerShell y ejecuta:
```
Get-Content "C:\Users\Personal\Documents\antigravity-ftmo-10k\logs\live_bot.log" -Wait
```

Mensajes importantes que veras:

| Mensaje | Significado |
|---------|------------|
| `[HEARTBEAT] equity=...` | Bot vivo, reporte cada 30 min |
| `[XAUUSD_h1] OPEN BUY 0.05L @ 2350` | Abrio una operacion |
| `News block: NFP` | Pausado por noticias de alto impacto |
| `DAILY HALT` | Perdida diaria alcanzada, retoma manana |
| `PROFIT LOCKED` | Objetivo FTMO alcanzado, para |
| `TOTAL DD HALT` | Perdida maxima, para permanentemente |

---

## FILTROS QUE APLICA EL BOT

**Noticias:** Bloquea nuevas entradas 30 minutos antes/despues de noticias de alto impacto (USD, EUR, JPY). Si hay posicion abierta, la cierra de forma defensiva.

**Spread:** No abre si el spread esta muy ancho:
- XAUUSD: max 50 puntos
- USDJPY: max 15 puntos
- EURUSD: max 10 puntos

**Una posicion por celda:** Nunca abre una segunda posicion si ya tiene una. El SL/TP la cierra automaticamente.

---

## ARCHIVOS IMPORTANTES

| Archivo | Para que sirve |
|---------|---------------|
| `live_bot.py` | El bot principal |
| `ARRANCAR_BOT.bat` | Doble click para arrancar |
| `bot_state.json` | Guarda el estado (balance inicial, dias operados) |
| `logs/live_bot.log` | Todo lo que hace el bot |
| `ff_calendar_thisweek.xml` | Calendario economico (se actualiza automaticamente) |

---

## PREGUNTAS FRECUENTES

**El bot arranco pero no abre operaciones**
Normal. Solo opera de 07:00-20:55, cuando hay señal valida, y si no hay noticias. Puede pasar horas sin operar.

**Veo "Spread too wide, skipping"**
El broker tiene spread elevado en ese momento (fin de semana, noticias). El bot lo ignora y sigue vigilando.

**Se cerro la ventana con un error**
Revisa `logs/live_bot.log`, busca la ultima linea. Luego vuelve a arrancar con el .bat — retoma donde lo dejo.

**Quiero saber el PnL actual sin revisar MT5**
Busca la ultima linea `[HEARTBEAT]` en el log. Aparece cada 30 minutos.

**El bot dijo "PROFIT LOCKED"**
Felicidades. El challenge esta ganado. Entra a la plataforma FTMO y solicita la cuenta fondeada.

---

## ARQUITECTURA (PARA REFERENCIA)

```
ARRANCAR_BOT.bat
  └── python live_bot.py
        ├── research/strategies/h1_trend_atr.py   (señales XAUUSD y USDJPY)
        ├── research/strategies/bb_squeeze.py      (señales EURUSD)
        └── MT5 API  →  broker  →  mercado
```

---

*Resultados backtested: 56.7% pass rate, 0% blown, en 300 simulaciones Monte Carlo de 30 dias.*

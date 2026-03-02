# Alternative Strategy Catalogue — Extension
### New Sandboxes, Commodity Driver Matrix, and the R-Multiple Compounding Engine
*v0.2 — March 2026*

---

## Part I: The Commodity Fundamental Driver Matrix

This is the idea you're most excited about, and rightly so — it's the single highest-value informational edge you can build. The concept: for every commodity you trade, maintain a **live dashboard of the 5-10 fundamental drivers** that actually move price, updated in real-time or daily. This turns you from a purely technical trader into a *technically-timed fundamental trader* — the exact profile of every great macro trader (PTJ, Druckenmiller, Bacon).

### Why This Works

CTL gives you *when* to enter. The driver matrix gives you **which setups to take and how much conviction to assign.** When a B1 fires on palladium AND you can see that South African mining output just dropped 8% AND auto catalyst demand is accelerating AND specs are still short per COT — that's a 4-layer confirmation stack. You size it at the top of your range. When a B1 fires but the fundamental picture is neutral or mixed, you take it at minimum size or skip it entirely.

This is exactly what Bacon called "information edge through network intelligence" — you're just systematizing it with data feeds instead of phone calls.

### The Matrix: Commodity by Commodity

---

#### PALLADIUM (PA) — Your Core Market

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| South African mine production | SA Minerals Council, Norilsk quarterly reports | Monthly/Quarterly | Supply shock |
| Russian export policy/sanctions | News feeds, OFAC updates, shipping data | Real-time | Supply shock |
| Global auto production (ICE vehicles) | IHS Markit, OICA, China CPCA | Monthly | Demand driver |
| EV adoption rate (substitution threat) | BloombergNEF, IEA monthly | Monthly | Long-term demand headwind |
| Platinum-palladium substitution ratio | Spread: PA-PL, WPIC reports | Daily | Relative value |
| Above-ground inventories (ETF holdings + Zurich/London vaults) | ETF Securities, LPPM | Daily/Weekly | Supply buffer |
| Recycling supply (autocatalyst scrappage) | Johnson Matthey, BASF reports | Quarterly | Secondary supply |
| China import volumes | China customs data via Bloomberg/Reuters | Monthly | Demand pull |
| Speculative positioning | COT report (CFTC) | Weekly (Tues data, Fri release) | Sentiment |
| USD strength (DXY) | Real-time market data | Real-time | Pricing factor |

**Key regime indicator:** The PA/PL spread. When palladium trades at a large premium to platinum, substitution risk rises. When the spread compresses, it often signals a palladium supply squeeze or platinum oversupply.

---

#### GOLD (GC)

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| Real interest rates (10Y TIPS yield) | FRED, real-time bond data | Real-time | Primary driver |
| Central bank buying (official reserves) | World Gold Council, IMF COFER | Monthly/Quarterly | Structural demand |
| USD index (DXY) | Real-time | Real-time | Inverse correlation |
| Geopolitical risk index | Caldara-Iacoviello GPR Index | Daily | Fear bid |
| ETF flows (GLD, IAU holdings) | ETF provider daily reports | Daily | Western investment demand |
| Shanghai Gold Exchange premium/discount | SGE data | Daily | Chinese physical demand |
| COMEX inventory + delivery notices | CME Group | Daily | Physical tightness |
| Mine production (top 5 producers) | World Gold Council, company reports | Quarterly | Supply trend |
| Speculative positioning | COT report | Weekly | Sentiment |
| Fed policy expectations (Fed funds futures) | CME FedWatch | Real-time | Rate path proxy |

**Key regime indicator:** Real yields. Gold's primary driver is the opportunity cost of holding a zero-yield asset. When real 10Y yields fall below 0%, gold enters a structural bull regime. Track TIPS yields as your north star.

---

#### CRUDE OIL (CL)

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| OPEC+ production quotas & compliance | OPEC MOMR, S&P Global Platts | Monthly + ad hoc | Supply management |
| US shale production (EIA weekly) | EIA Weekly Petroleum Status | Weekly (Wed) | Supply response |
| US crude inventories (Cushing + total) | EIA + API (Tues preview) | Weekly | Physical balance |
| Global floating storage (tanker tracking) | Kpler, Vortexa | Daily | Hidden inventory |
| China crude imports (teapot refinery runs) | China customs, Shandong data | Monthly | Marginal demand |
| Crack spreads (3-2-1, gasoline, distillate) | Real-time futures | Real-time | Refinery economics |
| Strategic Petroleum Reserve (SPR) levels | EIA | Weekly | Policy supply |
| Geopolitical risk (Strait of Hormuz, Libya, etc.) | News + shipping AIS data | Real-time | Supply disruption |
| Speculative positioning | COT + managed money | Weekly | Sentiment |
| Term structure (contango/backwardation) | Futures curve | Real-time | Physical tightness signal |

**Key regime indicator:** The Cushing inventory level + term structure combo. Low Cushing + backwardation = tight physical market = bullish. High Cushing + contango = glut = bearish. This is the single most reliable crude regime signal.

---

#### NATURAL GAS (NG)

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| EIA storage report (injection/withdrawal vs expectations) | EIA Natural Gas Weekly | Weekly (Thurs) | Primary short-term catalyst |
| Weather forecasts (HDD/CDD) | NOAA, DTN, private weather models | Daily | Demand driver |
| US production (Appalachia, Permian associated gas) | EIA monthly, Genscape/East Daley | Monthly/Weekly | Supply trend |
| LNG export volumes (feedgas to terminals) | EIA, Marine Traffic, Kpler | Daily | Demand pull |
| Henry Hub vs TTF/JKM spread (global arb) | Real-time futures | Real-time | Export economics |
| Power burn (gas-to-coal switching) | EIA, ERCOT/PJM data | Daily | Demand variability |
| Mexican pipeline exports | Genscape, CFE data | Weekly | Incremental demand |
| Hurricane season outlook + active storms | NHC, NOAA seasonal outlook | Seasonal + real-time | Production disruption |
| Speculative positioning | COT report | Weekly | Sentiment |
| Seasonality (5/10/20yr avg storage path) | EIA historical | Updated weekly | Context |

**Key regime indicator:** Storage vs. 5-year average. Below average = bullish lean. Above = bearish lean. The deviation from average entering winter (Oct-Nov) is the single best predictor of winter nat gas price.

---

#### COPPER (HG)

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| China PMI (manufacturing) | NBS, Caixin | Monthly | Demand proxy |
| LME/COMEX/Shanghai warehouse stocks | Exchange data | Daily | Physical balance |
| Chile/Peru mine production + disruptions | Cochilco, company reports | Monthly | Supply driver |
| China property construction (new starts, completion) | NBS, Mysteel | Monthly | Demand driver |
| Global EV production (copper intensity) | BloombergNEF, IEA | Monthly | Structural demand |
| Grid investment (copper wire demand) | IEA, State Grid Corp data | Quarterly | Green transition demand |
| TC/RC (smelter treatment charges) | Fastmarkets, CRU | Weekly | Concentrate supply signal |
| Scrap spread (No. 2 copper vs cathode) | Fastmarkets | Daily | Secondary supply |
| Speculative positioning | COT + LME positioning data | Weekly | Sentiment |
| USD + China credit impulse | Real-time + PBOC data | Real-time/Monthly | Macro overlay |

**Key regime indicator:** LME warehouse stocks + China credit impulse. Falling stocks + expanding credit = copper bull market. This combo has preceded every major copper rally in the last 20 years.

---

#### SILVER (SI)

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| Gold/silver ratio | Real-time (GC/SI) | Real-time | Relative value |
| Industrial demand (solar PV installations) | SEIA, BloombergNEF | Monthly/Quarterly | Structural demand |
| Mine production (Mexico, Peru, China) | Silver Institute, GFMS | Quarterly | Supply |
| ETF holdings (SLV, etc.) | ETF provider reports | Daily | Investment demand |
| India import demand | DGFT customs data | Monthly | Physical demand |
| COMEX inventories + delivery | CME Group | Daily | Physical tightness |
| Speculative positioning | COT report | Weekly | Sentiment |
| Electronics/5G demand | Industry reports | Quarterly | Industrial demand |
| Real yields (follows gold driver) | TIPS yields | Real-time | Macro driver |
| Recycling supply | Silver Institute | Annual | Supply context |

**Key regime indicator:** Gold/silver ratio. When it's >80, silver is historically cheap vs gold and tends to outperform in the next bull leg. When <60, silver has caught up. The ratio also signals risk appetite — silver outperforms in risk-on environments.

---

#### WHEAT (ZW) / CORN (ZC) / SOYBEANS (ZS) — Agriculture Complex

| Driver | Data Source | Update Freq | Signal Type |
|--------|-----------|-------------|-------------|
| USDA WASDE report (supply/demand balance) | USDA | Monthly (usually ~12th) | Primary fundamental |
| US crop conditions (good/excellent %) | USDA Weekly Crop Progress | Weekly (Mon, Apr-Nov) | Growing season health |
| Global export inspections + sales | USDA Weekly Export Inspections | Weekly (Mon) | Demand pull |
| Brazil/Argentina planting/harvest progress | CONAB, Buenos Aires Grain Exchange | Weekly/Bi-weekly | Southern hemisphere supply |
| Black Sea export volumes (Ukraine, Russia) | UkrAgroConsult, IKAR, shipping data | Weekly | Geopolitical supply |
| China import demand (soybean crush margins) | CNGOIC, Dalian futures | Daily/Weekly | Marginal demand |
| US planting intentions (USDA Prospective Plantings) | USDA | Annual (March 31) | Major catalyst |
| Weather (US Midwest, Brazil cerrado, Argentina pampas) | NOAA, private forecasters | Daily | Growing season risk |
| Ethanol blend margins (corn) | Real-time | Real-time | Industrial demand |
| Speculative positioning | COT report | Weekly | Sentiment |

**Key regime indicator:** USDA stocks-to-use ratio. Below 10% = tight market, price spikes likely. Above 15% = comfortable supply, price ceiling. The WASDE report day is the single most important fundamental catalyst — trade the reaction with CTL.

---

## Part II: New Strategies & Sandboxes

(Original content preserved from user-provided note.)

### 9. Crypto Perpetual Futures Funding Rate Arbitrage
### 10. Interest Rate Futures — Fed Policy Alpha
### 11. Equity Sector Rotation via Futures/ETFs
### 12. Volatility Surface Arbitrage (VIX/VVIX Structure)
### 13. Commodity Spread Trading (Inter- and Intra-Commodity)
### 14. Macro Event Straddles (Earnings, FOMC, OPEC, USDA)
### 15. Cross-Exchange and Cross-Instrument Basis Trades
### 16. Power and Emissions Markets (Frontier Sandbox)

---

## Part III: The R-Multiple Compounding Engine

(Original content preserved from user-provided note.)

---

## Part IV: Updated Strategy Matrix (Full Catalogue)

(Original content preserved from user-provided note.)

---

## Part V: The Integrated Architecture Vision

(Original content preserved from user-provided note.)

---

## Appendix: Quick-Start Checklist

(Original content preserved from user-provided note.)

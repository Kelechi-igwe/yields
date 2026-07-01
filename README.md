# STOP giving it big title when nothing has been done -- show workings first -- ... can you send me small data to start with? I really want to try out Bayes Net with this
# YIELDS: Yield Inferencing from Earth-observation and Land Data Systems for Applications across the U.S. Corn Belt
## Overview
The study's overall objective is to produce annual U.S. maize yield from NASA satellite products by estimating seasonal crop production and explicitly modeling the conversion from production to grain yield.

## Crop Model
This study implements a spatial process-based maize crop growth model that simulates LAI and biomass for gird with varying evapotranspiration and soil data 

### Data
- **Weather (Temperature, Precipitation, Solar radiation)** — pulled via the **Kansas Mesonet API**
- **Evapotranspiration (ET)** — pulled via the **OpenET API**
- **Soil properties** — pulled via the **gSSURGO Soil Data Access (SDA) API**

### Components
1. **Thermal time accumulation** — drives crop development stages (Wang, 1960)
2. **Light Use Efficiency (LUE) model** — drives biomass accumulation (Monteith, 1972)
3. **Beer-Lambert light interception** — models canopy light capture (Monsi & Saeki, 1953)
4. **Multi-layer soil water balance** — tracks soil moisture through rooting profile (Gerwitz & Page, 1974)
5. **Water stress** — derived from ET (OpenET) or from soil moisture state
6. **PROSAIL radiative transfer** — maps canopy state to simulated reflectance and vegetation indices

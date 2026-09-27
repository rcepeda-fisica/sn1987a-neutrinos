# SN1987A: los neutrinos de una supernova

Autor: Roberto Cepeda. Código y datos del póster de la asignatura *Física Nuclear y de Partículas Elementales* (Universidad de Oviedo, curso 2026-27).

Se reanalizan los 24 antineutrinos de la supernova SN1987A detectados el 23 de febrero de 1987 por Kamiokande-II, IMB y Baksan, para responder a dos preguntas:

1. ¿Cuánta energía se llevaron los neutrinos, y a qué temperatura se emitieron?
2. ¿Qué cota impone el tiempo de vuelo a la masa del neutrino?

## Contenido

| Archivo | Qué es |
|---|---|
| `sn1987a_eventos.csv` | Los 29 eventos publicados (16 KII, 8 IMB, 5 Baksan): tiempo relativo, energía, error, ángulo y tasa de fondo |
| `sn1987a_detectores.csv` | Parámetros de los tres detectores: protones blanco, umbral, resolución, fondo |
| `analisis_sn1987a.py` | Ajuste del espectro térmico y cota a la masa; genera las tres figuras del póster |
| `fig1_eventos.png` | Energía y tiempo de llegada de los 24 eventos |
| `fig2_ajuste_espectro.png` | Región permitida (T, W) al 68 % y 95 %, y espectro observado frente al ajuste |
| `fig3_masa_neutrino.png` | Retraso por masa frente a los datos y dispersión de la emisión inferida en función de m |

## Método

**Cálculo 1.** Fluencia térmica de antineutrinos electrónicos con dos parámetros (energía radiada W y temperatura T), sección eficaz de la desintegración β inversa, eficiencia y resolución de cada detector, y ajuste de máxima verosimilitud de Poisson sin binear a los 11 eventos de KII (E ≥ 7.5 MeV) y los 8 de IMB.

**Cálculo 2.** Retraso por masa Δt = (L/2c)(mc²/E)²; se exige que la dispersión por masa de los 11 eventos de KII no supere la duración observada del estallido.

## Resultados

- W = 4.4 (+1.6/−1.3)·10⁵² erg, T = 4.1 (+0.5/−0.4) MeV (68 % C.L.); con equipartición, W_total ≈ 2.6·10⁵³ erg, comparable a la energía de ligadura gravitatoria de una estrella de neutrones de 1.4 masas solares.
- m < 17 eV.

## Cómo ejecutarlo

```bash
pip install numpy scipy pandas matplotlib
python analisis_sn1987a.py
```

Imprime los resultados numéricos y guarda las tres figuras en el directorio actual.

## Fuentes de los datos

- K. Hirata et al. (Kamiokande-II), *Phys. Rev. Lett.* 58, 1490 (1987).
- R. M. Bionta et al. (IMB), *Phys. Rev. Lett.* 58, 1494 (1987).
- E. N. Alekseev et al. (Baksan), *Phys. Lett. B* 205, 209 (1988).
- Tabla de eventos y parámetros de los detectores tomados de la recopilación de F. Vissani, *J. Phys. G* 42, 013001 (2015).

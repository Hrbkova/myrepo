# Multidimensional Scaling of Open-Ended Political Identity Descriptions
## Czech Republic Survey Analysis

---

## Methods

### Text Preprocessing and Embedding

Open-ended responses describing political ingroups (n=1,396) and outgroups (n=1,382) were preprocessed using Czech-language lemmatization with UDPipe, retaining nouns, verbs, and adjectives. Responses containing only "don't know" statements or unintelligible text were removed (ingroup: 56 removed, 4.0%; outgroup: 83 removed, 6.0%), yielding final samples of 1,340 ingroup and 1,299 outgroup descriptions.

Each response was converted to a 384-dimensional vector using a multilingual sentence transformer model (paraphrase-multilingual-MiniLM-L12-v2), which captures semantic similarity across languages. Pairwise cosine distances were computed between all response embeddings, producing a distance matrix representing the semantic dissimilarity structure of the corpus.

### Multidimensional Scaling

We applied metric MDS to project responses into a lower-dimensional space while preserving pairwise distances. To determine the optimal dimensionality, we computed stress values for solutions ranging from 1 to 10 dimensions. Both ingroup and outgroup responses exhibited remarkably similar scree patterns, with stress declining sharply from 1 to 3 dimensions before leveling off:

| Dimensions | Ingroup Stress | Outgroup Stress | Reduction |
|------------|----------------|-----------------|-----------|
| 1          | 114,034        | 104,421         | —         |
| 2          | 40,308         | 36,811          | 65%       |
| 3          | 20,106         | 19,472          | 50%       |
| 4          | 12,475         | 11,646          | 38%       |
| 5          | 8,460          | 7,675           | 32%       |

We retained the 3-dimensional solution based on three criteria: (1) the scree plot "elbow" indicating diminishing returns after 3 dimensions; (2) substantive interpretability of each dimension; and (3) parsimony for cross-national comparison. A fourth dimension captured a diffuse mix of political disengagement and interpersonal warmth but lacked theoretical coherence.

The parallel dimensionality across ingroup and outgroup descriptions suggests that Czech citizens employ similar cognitive frameworks when characterizing both political allies and adversaries.

### Dimension Interpretation

We interpreted each dimension by examining responses at the extreme poles (±1 SD from the mean).

---

## Results: Ingroup Dimensions

Three distinct dimensions structure how Czechs construct political ingroup identities:

### Ingroup Dimension 1: Analytical/Educated vs. Ordinary Middle Class

The first dimension separates respondents who emphasize education, critical thinking, and information verification from those who emphasize ordinary middle-class identity based on income and work.

**LOW pole (Analytical/Educated):**
| Score | Response |
|-------|----------|
| -0.66 | "Lidi kteří pořad za vše platí zdražuje se" |
| -0.60 | "Realističtější, objektivní, sledující dění kolem sebe" |
| -0.56 | "Jsou to většinou vzdělaní, analytického myšlení schopní lidé, kteří si zdroje informací ověřuji a konfrontují" |
| -0.54 | "Overujú si fakty, vedia si dať dokopy dve a dve. Sú to ľudia rozmýšľajúci kriticky" |
| -0.53 | "Inteligentní, prozíravý, s dobrou pamětí" |

**HIGH pole (Ordinary Middle Class):**
| Score | Response |
|-------|----------|
| 0.53 | "Normální pracujíci lidé s průměrnou i nižší mzdou" |
| 0.59 | "Lidé co se chtějí mít dobře, nechtějí se dřít pro pár korun a většinu své výplaty dát státu na daně" |
| 0.59 | "Lidi co pracují a odvádějí daně" |
| 0.61 | "Pracující lidé, kteří nepotřebují, aby se o ně stát staral" |
| 0.65 | "Vlastenci, lidé kteří nechtějí zaprodat svou zem a být ve vlastní zemi otroky" |

**Interpretation:** This dimension reflects the education/class divide reshaping European politics. Educated Czechs define belonging through epistemic virtue ("we verify information") while middle-class Czechs emphasize economic identity ("we work, we pay taxes, we're ordinary people").

---

### Ingroup Dimension 2: Foreign Policy/Supranational vs. Tribal/Affective

The second dimension distinguishes respondents who articulate their identity through specific foreign policy positions from those using vague, tribal language.

**LOW pole (Foreign Policy/Supranational):**
| Score | Response |
|-------|----------|
| -0.58 | "Nesouhlasí s EU, s politikou evropské unie, nesouhlasí se zavedením eura. Nepodporují válku na Ukrajině" |
| -0.55 | "Odpůrci současné vlády, odpůrci migrace a Green Dealu" |
| -0.54 | "Vystudovaní ekonomové, kteří mají přehled, jaký je vývoj ekonomik v ostatních evropských státech" |
| -0.54 | "Jde jim o slušnou a reprezentativní vládu, která není xenofobní, homofobní" |
| -0.54 | "Ekonomický liberál, konzervativec, euroskeptik, thatcherista, antibolševik" |

**HIGH pole (Tribal/Affective):**
| Score | Response |
|-------|----------|
| 0.65 | "Takové neznám ale myslím že mají stejný názor" |
| 0.65 | "Stejná krevní skupina se stejnými názory" |
| 0.66 | "JSOU TO MOJI NEJLEPŠI KAMARADI" |
| 0.76 | "Jsou na stejné vlně jako já" |
| 0.78 | "Soukmenovci se systémem. Chtějí změnu a už mají tohoto hnusu dost" |

**Interpretation:** The LOW pole contains respondents from DIFFERENT political positions (both euroskeptics and pro-Europeans) but they all articulate SPECIFIC positions. The HIGH pole uses vague tribal language ("same wavelength", "my friends", "same blood type") without substantive political content. Notably, the articulated positions cluster around EU and foreign policy issues, suggesting that in the Czech context, European integration serves as the primary axis of political identity articulation.

---

### Ingroup Dimension 3: Belief-based vs. Social Position Identity

The third dimension separates those who define their group through political beliefs and values from those who use demographic and social descriptors.

**LOW pole (Belief-based):**
| Score | Response |
|-------|----------|
| -0.55 | "Prozápadní, podporující vstup do EU, souhlas s podporou Ukrajiny, odmítající KSČM, SPD a ANO" |
| -0.55 | "Radujeme se u stejných zpráv na internetu, dokážeme se rozčílit z rasistických keců, nebo urážek gay" |
| -0.52 | "Progresivní smýšlení bez šíření nenávisti, dezinformací a lží" |
| -0.51 | "Jsou to zpravidla pravičáci, ateisté, většinou inteligentní, zelenou ideologii, gender apod. vnímají kriticky" |
| -0.50 | "Nevěřící vnucované propagandě státu" |

**HIGH pole (Social Position):**
| Score | Response |
|-------|----------|
| 0.63 | "Člověk střední vrstvy, s normálním platem a prací na plný úvazek" |
| 0.63 | "Mladí lidé, studenti VŠ, těší se ze života" |
| 0.68 | "Jsou to normální lidi s průměrným příjmem" |
| 0.69 | "Jsou to vzdělaní lidé, minimálně s maturitou. Všichni pracují. Jsou to buď vedoucí pracovníci, manažeři" |
| 0.73 | "Vysokoškoláci či lidé s maturitním vzděláním s průměrnými či nprůměrnými příjmy" |

**Interpretation:** The LOW pole defines identity through WHAT the group believes (whether progressive or conservative values). The HIGH pole defines identity through WHO the group members are socially (class, education level, income, lifestyle). Note that the LOW pole is NOT ideologically consistent—it includes both progressives and conservatives—but all describe their group through beliefs rather than demographics.

---

## Results: Outgroup Dimensions

Two clearly interpretable dimensions structure how Czechs characterize political outgroups. A third dimension did not yield coherent interpretation.

### Outgroup Dimension 1: Substantive vs. Conformist Rejection

The first dimension captures HOW respondents characterize their outgroup: through specific descriptions or dismissive labels.

**LOW pole (Substantive):**
| Score | Response |
|-------|----------|
| -0.66 | "Nejvíce se mé názory liší od lidí důchodového věku s podprůměrnými příjmy, či lidí se základním vzděláním" |
| -0.65 | "Jsou příznivci Evropské unie, nevadí jim migrace, třídí odpad, jedou často eko-bio, smýšlejí pozitivně" |
| -0.64 | "Většinou jsou to lidé kteří mají jen základní vzdělání a jsou okolo 50 let a poté ještě lidé přes 70" |
| -0.60 | "Žádné nebo minimální vzdělání, nízký příjem, věčně něco kritizující, aniž by sami se zasloužili o zlepšení" |
| -0.58 | "Lidé, kteří chtějí zlikvidovat svými postoji a svými kroky tradiční nastavení křesťanské Evropy" |

**HIGH pole (Conformist):**
| Score | Response |
|-------|----------|
| 0.55 | "Lidé, kteří nepotřebují o ničem důkazy" |
| 0.56 | "Lidé, kteří věří všemu, co slyší v televizi a rádiu" |
| 0.61 | "Sledují ČT, nehledají jiné zdroje informací" |
| 0.65 | "Kam vítr, tam plášť" |
| 0.67 | "Takové ovce, co věří všemu ve zprávách" |
| 0.70 | "Ovce, co jdou s proudem a poslouchají lži" |

**Interpretation:** The LOW pole contains SPECIFIC descriptions of outgroups—whether demographic (age, education, income) or political (pro-EU liberals, traditionalists). The HIGH pole dismisses opponents as "sheep" (ovce) who lack independent thought—people "without their own position" (bez vlastního názoru). This dimension captures cognitive engagement with the outgroup: substantive critique vs. conformist dismissal.

---

### Outgroup Dimension 2: Extremist vs. Elite Threat

The second dimension captures WHAT TYPE of threat the outgroup represents: cultural/extremist or economic/elite.

**LOW pole (Extremist Threat):**
| Score | Response |
|-------|----------|
| -0.56 | "Lidé, kteří nepovažují globální oteplování za problém, kteří jsou proti LGBT komunitě, kteří šíří dezinformace" |
| -0.55 | "Extremistické až rasistické názory. Nedokončené základní vzdělání" |
| -0.55 | "Neinformovaní o situaci v Česku, naopak ovlivněni fake news a dezinformacemi" |
| -0.53 | "Nemají toleranci vůči LGBTQ, soudí lidi podle jejich orientace, vzhledu" |
| -0.52 | "Rasista, neonacista... lidé, kteří neposlouchají cizí argumenty a nechtějí se dohodnout na kompromisu" |

**HIGH pole (Elite Threat):**
| Score | Response |
|-------|----------|
| 0.62 | "Většinou jsou to zbohatlíci a komedianti z divadel" |
| 0.66 | "Velcí podnikatelé, zbohatlíci" |
| 0.67 | "Staré struktury, které by rády nastolily přežité doby" |
| 0.68 | "Většinou lidé s velkým platem a velkého města" |
| 0.69 | "Zbohatlíci, velcí podnikatelé, manažeři, ředitelé" |
| 0.73 | "Bohatí lidé a majitelé firem" |

**Interpretation:** This dimension reflects the classic two-dimensional space of political conflict. The LOW pole perceives a CULTURAL/EPISTEMIC threat: extremists, racists, intolerant people, disinformation spreaders—people with WRONG and RIGID positions. The HIGH pole perceives an ECONOMIC threat: wealthy elites, businessmen, "zbohatlíci" (nouveau riche). This maps onto the distinction between cultural and economic dimensions of political competition.

---

### Outgroup Dimension 3: Not Clearly Interpretable

The third outgroup dimension did not yield a coherent interpretation. The LOW pole mixed respondents who reject pro-government progressives with those who reject anti-government conspiracists. The HIGH pole contained a heterogeneous mix of conformist dismissals, economic critiques, and non-responses. We therefore report only two interpretable outgroup dimensions.

---

## Summary Table

### Ingroup Dimensions

| Dimension | Label | LOW Pole | HIGH Pole |
|-----------|-------|----------|-----------|
| 1 | **Analytical vs. Ordinary** | Educated, verify information, critical thinking | Middle class, average income, "normal people" |
| 2 | **Foreign Policy vs. Tribal** | Specific EU/Ukraine/migration positions | "Same wavelength", friends, tribal belonging |
| 3 | **Belief-based vs. Social Position** | Defined by values and political positions | Defined by class, education, demographics |

### Outgroup Dimensions

| Dimension | Label | LOW Pole | HIGH Pole |
|-----------|-------|----------|-----------|
| 1 | **Substantive vs. Conformist** | Specific descriptions (demographic or political) | "Sheep", no own position, followers |
| 2 | **Extremist vs. Elite Threat** | Cultural threat (racist, intolerant, disinformation) | Economic threat (rich, businessmen, elite) |

---

## Discussion Points

1. **Education/Class Cleavage:** Ingroup Dimension 1 reflects the diploma divide reshaping European politics—a divide between those who define belonging through epistemic virtue versus economic identity.

2. **EU as Primary Cleavage:** Ingroup Dimension 2 shows that politically articulate Czechs define themselves through EU and foreign policy positions, suggesting European integration is the primary structuring divide.

3. **Two Types of Epistemic Failure:** The outgroup dimensions reveal two distinct types of epistemic critique:
   - **Conformist** (Dim 1 HIGH): "They have no position of their own, they're sheep" — PASSIVE epistemic failure
   - **Extremist** (Dim 2 LOW): "They're racist, intolerant, won't listen to arguments" — ACTIVE epistemic failure

4. **Cultural vs. Economic Threat:** Outgroup Dimension 2 captures the two-dimensional threat perception: some Czechs fear cultural/extremist threats while others fear economic/elite threats.

---

## Next Steps

1. **Regression Analysis:** Test whether these dimension scores predict affective polarization (sympathy_gap, compromise_gap)

2. **Cross-National Comparison:** Replicate analysis in Hungary and Lithuania to assess generalizability

3. **Partisan Breakdown:** Examine whether dimension scores vary by vote choice (ANO, ODS, Piráti, SPD, etc.)

---

*Generated from MDS analysis of Czech survey data (n=1,437)*

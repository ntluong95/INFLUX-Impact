"""
build_animal_diseases_merged.py
--------------------------------
Integrates two animal disease datasets:
  - data/inputs/animal_diseases_woah_raw.csv   (207 rows; disease names, animal groups)
  - data/inputs/animal_diseases_raw.xlsx       (40 rows; pathogen names, subtypes, aka)

Output: data/inputs/animal_diseases.xlsx
Columns: pathogen_scientific_name_en | pathogen_subtypes_names_en | aka_en | disease_name_en

Domain-expert knowledge is embedded in WOAH_PATHOGEN_MAP to fill scientific names
for WOAH entries not already covered by the XLSX seed data.
"""

from __future__ import annotations
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WOAH_CSV  = ROOT / "data/inputs/animal_diseases_woah_raw.csv"
RAW_XLSX  = ROOT / "data/inputs/animal_diseases_raw.xlsx"
OUT_XLSX  = ROOT / "data/inputs/animal_diseases.xlsx"

# ---------------------------------------------------------------------------
# Expert mapping: WOAH disease_name -> (scientific_name, subtypes, aka, canonical_disease)
# Keys are normalised to lower-case, stripped.
# Values: (pathogen_scientific_name_en, pathogen_subtypes_names_en, aka_en, disease_name_en)
# Any value may be None / empty string -> will be left blank.
# ---------------------------------------------------------------------------
WOAH_PATHOGEN_MAP: dict[str, tuple[str, str, str, str]] = {
    # ── Aquatic ──────────────────────────────────────────────────────────────
    "abalone viral ganglioneuritis (abalone herpesvirus)": (
        "Aurivirus haliotidmalaco1", "", "Haliotid herpesvirus 1 (HaHV-1)", "Abalone viral ganglioneuritis"),
    "abalone viral ganglioneuritis": (
        "Aurivirus haliotidmalaco1", "", "Haliotid herpesvirus 1 (HaHV-1)", "Abalone viral ganglioneuritis"),
    "acute hepatopancreatic necrosis disease": (
        "Vibrio parahaemolyticus", "", "AHPND", "Acute hepatopancreatic necrosis disease"),
    "bonamiosis (b. exitiosa)": (
        "Bonamia exitiosa", "", "", "Bonamiosis"),
    "bonamiosis (b. ostreae)": (
        "Bonamia ostreae", "", "", "Bonamiosis"),
    "carp edema virus": (
        "Carp edema virus", "", "CEV", "Carp edema"),
    "chytridiomycosis (batrachochytrium dendrobatidis)": (
        "Batrachochytrium dendrobatidis", "", "Bd", "Chytridiomycosis"),
    "chytridiomycosis (batrachochytrium salmandrivorans)": (
        "Batrachochytrium salamandrivorans", "", "Bsal", "Chytridiomycosis"),
    "crayfish plague": (
        "Aphanomyces astaci", "", "", "Crayfish plague"),
    "decapod iridescent virus 1": (
        "Decapod iridescent virus 1", "Shrimp hemocyte iridescent virus",
        "Shrimp hemocyte iridescent virus", "Decapod iridescent virus 1 disease"),
    "epizootic haematopoietic necrosis disease": (
        "Epizootic haematopoietic necrosis virus", "", "EHNV",
        "Epizootic haematopoietic necrosis disease"),
    "epizootic ulcerative syndrome": (
        "Aphanomyces invadans", "", "A. piscicida (syn.)", "Epizootic ulcerative syndrome"),
    "gyrodactylosis (g. salaris)": (
        "Gyrodactylus salaris", "", "", "Gyrodactylosis"),
    "infection with enterocytozoon hepatopenaei": (
        "Enterocytozoon hepatopenaei", "", "EHP",
        "Infection with Enterocytozoon hepatopenaei"),
    "infection with covert mortality nodavirus (cmnv)": (
        "Covert mortality nodavirus", "", "CMNV",
        "Infection with covert mortality nodavirus"),
    "infection with infectious salmon anaemia virus": (
        "Infectious salmon anemia virus", "Infectious salmon anemia virus",
        "Salmon isavirus, ISAV", "Infectious salmon anaemia"),
    "infection with tilapia lake virus (tilv)": (
        "Tilapinevirus tilapiae", "Tilapia lake virus", "",
        "Tilapia lake virus disease"),
    "infectious haematopoietic necrosis": (
        "Novirhabdovirus salmonid",
        "Infectious hematopoietic necrosis virus",
        "Salmonid novirhabdovirus, IHNV",
        "Infectious haematopoietic necrosis"),
    "infectious hypodermal and haematopoietic necrosis": (
        "Penstylhamaparvovirus decapod1", "", "PstDV1, IHHNV",
        "Infectious hypodermal and haematopoietic necrosis"),
    "infectious myonecrosis": (
        "Penaeid shrimp infectious myonecrosis virus", "", "IMNV",
        "Infectious myonecrosis"),
    "koi herpesvirus disease": (
        "Cyvirus cyprinidallo3", "Cyprinid herpesvirus 3",
        "Koi herpesvirus (KHV)", "Koi herpesvirus disease"),
    "marteiliosis (m. refringens)": (
        "Marteilia refringens", "", "", "Marteiliosis"),
    "megalocytivirus pagrus1": (
        "Megalocytivirus pagrus1", "Red seabream iridovirus",
        "RSIV", "Megalocytivirus disease"),
    "necrotising hepatopancreatitis (hepatobacter penaei)": (
        "Candidatus Hepatobacter penaei", "", "NHP bacterium",
        "Necrotising hepatopancreatitis"),
    "perkinsosis (p. marinus)": (
        "Perkinsus marinus", "", "", "Perkinsosis"),
    "perkinsosis (p. olseni)": (
        "Perkinsus olseni", "", "", "Perkinsosis"),
    "ranavirosis": (
        "Ranavirus spp.", "", "", "Ranavirosis"),
    "salmonid alphavirus infection": (
        "Salmonid alphavirus", "", "SAV", "Salmonid alphavirus infection"),
    "spring viraemia of carp": (
        "Sprivivirus cyprinus", "", "Spring viraemia of carp virus",
        "Spring viraemia of carp"),
    "taura syndrome": (
        "Aparavirus tauraense", "Taura syndrome virus",
        "Taura syndrome virus, TSV", "Taura syndrome"),
    "viral haemorrhagic septicaemia": (
        "Novirhabdovirus piscine", "Viral hemorrhagic septicemia virus",
        "Viral hemorrhagic septicemia virus, VHSV",
        "Viral haemorrhagic septicaemia"),
    "white spot disease": (
        "White spot syndrome virus", "", "WSSV", "White spot disease"),
    "white tail disease": (
        "Macrobrachium rosenbergii nodavirus", "", "MrNV",
        "White tail disease"),
    "withering abalone syndrome (xenohaliotis californiensis)": (
        "Candidatus Xenohaliotis californiensis", "",
        "Xenohaliotis californiensis", "Withering abalone syndrome"),
    "withering abalone syndrome": (
        "Candidatus Xenohaliotis californiensis", "",
        "Xenohaliotis californiensis", "Withering abalone syndrome"),
    "yellow head disease": (
        "Okavirus flavicapitis", "Yellow head virus",
        "Gill-associated virus genotype 1, YHV",
        "Yellow head disease"),
    # ── Terrestrial ──────────────────────────────────────────────────────────
    "african horse sickness": (
        "African horse sickness virus", "9 serotypes (1-9)",
        "AHSV", "African horse sickness"),
    "african swine fever": (
        "African swine fever virus", "", "ASFV", "African swine fever"),
    "agent causing chronic wasting disease (cwd)": (
        "Chronic wasting disease prion", "", "CWD prion",
        "Chronic wasting disease"),
    "algal toxicosis": (
        "", "", "", "Algal toxicosis"),
    "anthrax": (
        "Bacillus anthracis", "", "", "Anthrax"),
    "atrophic rhinitis of swine": (
        "Bordetella bronchiseptica", "Pasteurella multocida toxigenic strains",
        "", "Atrophic rhinitis of swine"),
    "aujeszky's disease": (
        "Varicellovirus suidalpha1", "Suid alphaherpesvirus 1",
        "Aujeszky's disease virus, Pseudorabies virus, PRV",
        "Aujeszky's disease"),
    "avian influenza": (
        "Influenza A virus", "H5N1, H5N2, H5N8, H7N1, H7N3, H7N7, H7N9",
        "Highly pathogenic avian influenza (HPAI)", "Avian influenza"),
    "avian chlamydiosis": (
        "Chlamydia psittaci", "", "Chlamydiosis, Psittacosis",
        "Avian chlamydiosis"),
    "avian infectious bronchitis": (
        "Gammacoronavirus gallus", "", "IBV, Infectious bronchitis virus",
        "Avian infectious bronchitis"),
    "avian infectious laryngotracheitis": (
        "Iltovirus gallid2", "", "Gallid alphaherpesvirus 1, ILT virus",
        "Avian infectious laryngotracheitis"),
    "avian mycoplasmosis (mycoplasma gallisepticum)": (
        "Mycoplasma gallisepticum", "", "MG",
        "Avian mycoplasmosis"),
    "avian mycoplasmosis (mycoplasma synoviae)": (
        "Mycoplasma synoviae", "", "MS",
        "Avian mycoplasmosis"),
    "avian tuberculosis": (
        "Mycobacterium avium", "Mycobacterium avium subsp. avium",
        "Avian mycobacteriosis", "Avian tuberculosis"),
    "babesiosis (new or unusual occurrences)": (
        "Babesia spp.", "", "", "Babesiosis"),
    "baylisascariasis (b. procyonis)": (
        "Baylisascaris procyonis", "", "", "Baylisascariasis"),
    "bluetongue": (
        "Bluetongue virus", "28 serotypes (1-28)",
        "BTV", "Bluetongue"),
    "border disease": (
        "Border disease virus", "", "BDV, Pestivirus D",
        "Border disease"),
    "borreliasis": (
        "Borrelia spp.", "Borrelia burgdorferi, B. afzelii, B. garinii",
        "Lyme borreliosis", "Borreliasis"),
    "botulism": (
        "Clostridium botulinum", "Types A, B, C, D, E, F, G",
        "", "Botulism"),
    "bovine anaplasmosis": (
        "Anaplasma marginale", "", "", "Bovine anaplasmosis"),
    "bovine babesiosis": (
        "Babesia bovis", "Babesia bigemina, Babesia divergens",
        "", "Bovine babesiosis"),
    "bovine cysticercosis (c. jejuni and c. coli)": (
        "Taenia saginata", "", "Cysticercus bovis",
        "Bovine cysticercosis"),
    "bovine genital campylobacteriosis": (
        "Campylobacter fetus", "Campylobacter fetus subsp. venerealis",
        "", "Bovine genital campylobacteriosis"),
    "bovine spongiform encephalopathy": (
        "BSE prion", "", "Mad cow disease, BSE",
        "Bovine spongiform encephalopathy"),
    "bovine viral diarrhoea": (
        "Pestivirus bovis", "BVDV-1, BVDV-2",
        "Bovine viral diarrhea virus, BVDV",
        "Bovine viral diarrhoea"),
    "brucellosis": (
        "Brucella spp.",
        "Brucella abortus, B. melitensis, B. suis, B. canis, B. ovis",
        "", "Brucellosis"),
    "bunyaviral diseases of animals (excluding rift valley fever and crimean-congo haemorrhagic fever)": (
        "Phlebovirus spp.", "Schmallenberg virus, Akabane virus, Cache Valley virus",
        "Bunyaviridae", "Bunyaviral diseases of animals"),
    "camelpox": (
        "Camelpox virus", "", "CMLV", "Camelpox"),
    "campylobacteriosis (c. jejuni and c. coli)": (
        "Campylobacter jejuni", "Campylobacter coli",
        "", "Campylobacteriosis"),
    "caprine arthritis/encephalitis": (
        "Caprine arthritis encephalitis virus", "",
        "CAEV, Small ruminant lentivirus",
        "Caprine arthritis/encephalitis"),
    "chemical poisoning": (
        "", "", "", "Chemical poisoning"),
    "circovirosis": (
        "Porcine circovirus 2", "PCV2, PCV3",
        "Porcine circovirus-associated disease, PCVAD",
        "Circovirosis"),
    "classical swine fever": (
        "Classical swine fever virus", "",
        "Hog cholera virus, CSFV",
        "Classical swine fever"),
    "contagious agalactia": (
        "Mycoplasma agalactiae", "Mycoplasma mycoides, M. putrefaciens, M. capricolum",
        "", "Contagious agalactia"),
    "contagious bovine pleuropneumonia": (
        "Mycoplasma mycoides subsp. mycoides", "",
        "MmmSC", "Contagious bovine pleuropneumonia"),
    "contagious caprine pleuropneumonia": (
        "Mycoplasma capricolum subsp. capripneumoniae", "",
        "Mccp", "Contagious caprine pleuropneumonia"),
    "contagious equine metritis": (
        "Taylorella equigenitalis", "", "CEM organism",
        "Contagious equine metritis"),
    "crimean congo haemorrhagic fever": (
        "Crimean-Congo hemorrhagic fever orthonairovirus", "",
        "CCHFV", "Crimean-Congo haemorrhagic fever"),
    "cryptosporidiosis": (
        "Cryptosporidium spp.", "Cryptosporidium parvum, C. hominis",
        "", "Cryptosporidiosis"),
    "cysticercosis": (
        "Taenia spp.", "Taenia solium, Taenia saginata, Taenia multiceps",
        "", "Cysticercosis"),
    "diseases of bees": (
        "Multiple pathogens",
        "Varroa destructor, Aethina tumida, American foulbrood, European foulbrood",
        "Varroosis, Tropilaelaps infestation, Foulbrood",
        "Diseases of bees"),
    "dourine": (
        "Trypanosoma equiperdum", "", "", "Dourine"),
    "duck virus enteritis": (
        "Anatid alphaherpesvirus 1", "", "Duck herpesvirus 1, Duck plague virus",
        "Duck virus enteritis"),
    "duck virus hepatitis": (
        "Duck hepatitis A virus", "DHAV-1, DHAV-2, DHAV-3",
        "Duck hepatitis virus", "Duck virus hepatitis"),
    "ebola virus disease": (
        "Orthoebolavirus spp.",
        "Orthoebolavirus zairense, Orthoebolavirus sudanense",
        "Ebola virus, EBOV", "Ebola virus disease"),
    "echinococcosis": (
        "Echinococcus spp.",
        "Echinococcus granulosus, Echinococcus multilocularis",
        "Hydatidosis, Hydatid disease", "Echinococcosis"),
    "elephant endotheliotropic herpesvirus (eehv)": (
        "Proboscivirus elephantid1", "EEHV1A, EEHV1B, EEHV4, EEHV5",
        "Elephant endotheliotropic herpesvirus",
        "Elephant endotheliotropic herpesvirus disease"),
    "encephalomyocarditis virus infection": (
        "Encephalomyocarditis virus", "", "EMCV, Cardiovirus A",
        "Encephalomyocarditis"),
    "enzootic abortion of ewes": (
        "Chlamydia abortus", "", "Enzootic abortion, EAE, Ovine enzootic abortion",
        "Enzootic abortion of ewes"),
    "enzootic bovine leukosis": (
        "Bovine leukemia virus", "", "BLV, Bovine leukaemia virus",
        "Enzootic bovine leukosis"),
    "epizootic haemorrhagic disease": (
        "Epizootic hemorrhagic disease virus", "8 serotypes (1-8)", "EHDV",
        "Epizootic haemorrhagic disease"),
    "epizootic lymphangitis": (
        "Histoplasma farciminosum", "",
        "Cryptococcus farciminosus, Zymonema farciminosum",
        "Epizootic lymphangitis"),
    "equine encephalomyelitis (eastern)": (
        "Eastern equine encephalitis virus", "", "EEEV, EEE virus",
        "Equine encephalomyelitis (Eastern)"),
    "equine encephalomyelitis (western)": (
        "Western equine encephalitis virus", "", "WEEV, WEE virus",
        "Equine encephalomyelitis (Western)"),
    "equine infectious anaemia": (
        "Equine infectious anemia virus", "",
        "EIAV, Swamp fever virus",
        "Equine infectious anaemia"),
    "equine influenza": (
        "Influenza A virus", "H3N8, H7N7",
        "Equine influenza virus, EIV",
        "Equine influenza"),
    "equine piroplasmosis": (
        "Theileria equi", "Babesia caballi",
        "Equine babesiosis, Piroplasmosis",
        "Equine piroplasmosis"),
    "equine viral arteritis": (
        "Equine arteritis virus", "", "EAV, Equine arterivirus",
        "Equine viral arteritis"),
    "equine viral rhinopneumonitis (caused by ehv-1)": (
        "Equid alphaherpesvirus 1", "", "EHV-1, Equine herpesvirus 1",
        "Equine viral rhinopneumonitis"),
    "european brown hare syndrome virus": (
        "Lagovirus europaeus", "RHDV2 (GI.2)", "EBHSV",
        "European brown hare syndrome"),
    "fasciolosis (f. gigantica)": (
        "Fasciola gigantica", "", "", "Fasciolosis"),
    "fasciolosis (f. magna)": (
        "Fascioloides magna", "", "", "Fasciolosis"),
    "feline leukaemia": (
        "Feline leukemia virus", "", "FeLV", "Feline leukaemia"),
    "fibropapillomatosis (in sea turtles)": (
        "Chelonid fibropapilloma-associated herpesvirus", "",
        "FP-associated herpesvirus, CFPHV",
        "Fibropapillomatosis"),
    "filovirosis": (
        "Orthoebolavirus spp.", "Marburgvirus, Cuevavirus",
        "Ebola virus, Marburg virus",
        "Filovirosis"),
    "foot and mouth disease": (
        "Foot-and-mouth disease virus", "Serotypes O, A, C, SAT1, SAT2, SAT3, Asia1",
        "FMDV, Aphthovirus", "Foot and mouth disease"),
    "fowl cholera": (
        "Pasteurella multocida", "Serotypes A, B, D, E, F",
        "Avian cholera, Avian pasteurellosis",
        "Fowl cholera"),
    "fowl pox": (
        "Fowlpox virus", "", "Avipoxvirus, FPV", "Fowl pox"),
    "fowl typhoid": (
        "Salmonella enterica", "Salmonella Gallinarum",
        "Salmonella gallinarum",
        "Fowl typhoid"),
    "glanders": (
        "Burkholderia mallei", "", "Malleus",
        "Glanders"),
    "haemorrhagic septicaemia": (
        "Pasteurella multocida", "Serotype B:2, E:2",
        "Bovine pasteurellosis",
        "Haemorrhagic septicaemia"),
    "hantavirosis": (
        "Orthohantavirus spp.",
        "Sin Nombre orthohantavirus, Hantaan orthohantavirus, Seoul orthohantavirus",
        "Hantavirus, HCPS, HFRS",
        "Hantavirosis"),
    "heartwater": (
        "Ehrlichia ruminantium", "", "Cowdria ruminantium",
        "Heartwater"),
    "hendra virus (hev) infection": (
        "Henipavirus hendraense", "", "HeV",
        "Hendra virus infection"),
    "hendra virus": (
        "Henipavirus hendraense", "", "HeV",
        "Hendra virus infection"),
    "herpesvirosis (infection with alcelaphine herpesvirus 1 or ovine herpesvirus)": (
        "Alcelaphine gammaherpesvirus 1",
        "Ovine gammaherpesvirus 2",
        "Malignant catarrhal fever virus",
        "Malignant catarrhal fever"),
    "high pathogenicity avian influenza (hpai) in cattle": (
        "Influenza A virus", "H5N1 clade 2.3.4.4b",
        "HPAI H5N1",
        "High pathogenicity avian influenza in cattle"),
    "immunodeficiency virus infection": (
        "Primate lentivirus", "SIV, FIV",
        "Simian immunodeficiency virus, Feline immunodeficiency virus",
        "Immunodeficiency virus infection"),
    "infection with flavivirus (causing louping ill) – wild animals": (
        "Louping ill virus", "", "Tick-borne encephalitis virus (LI subtype)",
        "Louping ill"),
    "infection with influenza a viruses of high pathogenicity in birds other than poultry including wild birds": (
        "Influenza A virus", "H5N1, H5N8, H7N1",
        "HPAI in wild birds",
        "Highly pathogenic avian influenza in wild birds"),
    "infectious bovine rhinotracheitis/infectious pustular vulvovaginitis": (
        "Bovine alphaherpesvirus 1", "",
        "IBR/IPV, BHV-1",
        "Infectious bovine rhinotracheitis/infectious pustular vulvovaginitis"),
    "infectious bursal disease (gumboro disease)": (
        "Avibirnavirus gallinae", "", "IBDV, Gumboro disease virus",
        "Infectious bursal disease"),
    "influenza a virus of swine": (
        "Influenza A virus", "H1N1, H1N2, H3N2",
        "Swine influenza virus, SIV",
        "Influenza A virus of swine"),
    "japanese encephalitis": (
        "Japanese encephalitis virus", "", "JEV",
        "Japanese encephalitis"),
    "leishmaniosis": (
        "Leishmania spp.",
        "Leishmania donovani, L. infantum, L. braziliensis, L. major, L. tropica",
        "Leishmaniasis", "Leishmaniosis"),
    "leptospirosis": (
        "Leptospira spp.",
        "Leptospira interrogans, L. borgpetersenii, L. kirschneri",
        "", "Leptospirosis"),
    "leptospirosis (l. interrogans)": (
        "Leptospira interrogans", "", "", "Leptospirosis"),
    "listeriosis (l. monocytogenes)": (
        "Listeria monocytogenes", "", "", "Listeriosis"),
    "listeriosis (l. monocytogenes) in wild animals": (
        "Listeria monocytogenes", "", "", "Listeriosis"),
    "low pathogenic avian influenza (all subtypes)": (
        "Influenza A virus", "Multiple LPAI subtypes",
        "LPAI", "Low pathogenic avian influenza"),
    "lumpy skin disease": (
        "Lumpy skin disease virus", "", "LSDV, Capripoxvirus bovis",
        "Lumpy skin disease"),
    "mpox": (
        "Orthopoxvirus simiae", "", "Monkeypox virus, MPXV",
        "Mpox"),
    "maedi-visna": (
        "Maedi-visna virus", "", "MVV, Small ruminant lentivirus, Ovine progressive pneumonia virus",
        "Maedi-visna"),
    "malaria": (
        "Plasmodium spp.", "Plasmodium falciparum, P. vivax, P. malariae, P. ovale",
        "", "Malaria"),
    "malignant catarrhal fever": (
        "Alcelaphine gammaherpesvirus 1", "Ovine gammaherpesvirus 2",
        "Malignant catarrhal fever virus, MCF virus",
        "Malignant catarrhal fever"),
    "mammalian tuberculosis": (
        "Mycobacterium bovis", "Mycobacterium tuberculosis complex",
        "Bovine tuberculosis, bTB",
        "Mammalian tuberculosis"),
    "mange": (
        "Sarcoptes scabiei", "Demodex spp., Psoroptes spp.",
        "Scabies, Sarcoptic mange",
        "Mange"),
    "marek's disease": (
        "Gallid alphaherpesvirus 2", "",
        "Marek's disease virus, MDV",
        "Marek's disease"),
    "middle east respiratory syndrome (mers)": (
        "Middle East respiratory syndrome-related coronavirus", "",
        "MERS-CoV, Betacoronavirus 1",
        "Middle East respiratory syndrome"),
    "morbillivirosis (canids and felids)": (
        "Canine morbillivirus", "Feline morbillivirus",
        "Canine distemper virus, CDV",
        "Morbillivirosis"),
    "morbillivirosis (marine mammals)": (
        "Phocine morbillivirus", "Cetacean morbillivirus",
        "Phocine distemper virus, PDV",
        "Morbillivirosis"),
    "morbillivirosis in non-human primates": (
        "Measles morbillivirus", "",
        "Measles virus, MeV",
        "Morbillivirosis in non-human primates"),
    "morbillivirosis in other taxonomic groups of hosts": (
        "Morbillivirus spp.", "", "", "Morbillivirosis"),
    "mycotoxicosis": (
        "", "", "", "Mycotoxicosis"),
    "myxomatosis": (
        "Leporipoxvirus myxoma", "", "Myxoma virus, MYXV",
        "Myxomatosis"),
    "nairobi sheep disease": (
        "Nairovirus spp.", "Dugbe nairovirus, Ganjam virus",
        "Nairobi sheep disease virus, NSDV",
        "Nairobi sheep disease"),
    "new world screwworm (cochliomyia hominivorax)": (
        "Cochliomyia hominivorax", "", "",
        "New world screwworm myiasis"),
    "newcastle disease": (
        "Orthoavulavirus javaense", "",
        "Newcastle disease virus, NDV, Avian paramyxovirus 1",
        "Newcastle disease"),
    "newcastle disease (wild birds)": (
        "Orthoavulavirus javaense", "",
        "Newcastle disease virus, NDV",
        "Newcastle disease"),
    "nipah virus": (
        "Nipah henipavirus", "", "NiV",
        "Nipah virus infection"),
    "nipah virus infection (wild animals)": (
        "Nipah henipavirus", "", "NiV",
        "Nipah virus infection"),
    "nosemosis of honey bees": (
        "Vairimorpha apis", "Vairimorpha ceranae",
        "Nosema apis, Nosema ceranae",
        "Nosemosis of honey bees"),
    "old world screwworm (chrysomya bezziana)": (
        "Chrysomya bezziana", "", "",
        "Old world screwworm myiasis"),
    "ovine chlamydiosis": (
        "Chlamydia abortus", "", "Enzootic abortion of ewes",
        "Ovine chlamydiosis"),
    "ovine epididymitis (brucella ovis)": (
        "Brucella ovis", "", "",
        "Ovine epididymitis"),
    "ovine pulmonary adenocarcinoma (adenomatosis)": (
        "Jaagsiekte sheep retrovirus", "",
        "JSRV, Sheep pulmonary adenocarcinoma virus",
        "Ovine pulmonary adenocarcinoma"),
    "papillomatosis (in crocodiles)": (
        "Papillomavirus spp.", "", "",
        "Papillomatosis"),
    "paramyxovirosis (other than those listed by the woah)": (
        "Paramyxovirus spp.", "", "",
        "Paramyxovirosis"),
    "paratuberculosis": (
        "Mycobacterium avium subsp. paratuberculosis", "",
        "MAP, Johne's disease",
        "Paratuberculosis"),
    "parvovirosis": (
        "Parvovirus spp.",
        "Carnivore parvovirus, Canine parvovirus, Feline panleukopenia virus",
        "CPV, FPV",
        "Parvovirosis"),
    "pasteurellosis": (
        "Pasteurella multocida", "Mannheimia haemolytica",
        "", "Pasteurellosis"),
    "peste des petits ruminants": (
        "Morbillivirus caprinae", "",
        "PPR virus, PPRV",
        "Peste des petits ruminants"),
    "porcine cysticercosis": (
        "Taenia solium", "", "Cysticercus cellulosae",
        "Porcine cysticercosis"),
    "porcine epidemic diarrhoea": (
        "Porcine epidemic diarrhea virus", "",
        "PEDV, Alphacoronavirus 1",
        "Porcine epidemic diarrhoea"),
    "porcine reproductive and respiratory syndrome": (
        "Betaarterivirus suid 1", "PRRSV-1, PRRSV-2",
        "PRRSV, Blue-ear pig disease",
        "Porcine reproductive and respiratory syndrome"),
    "pox viruses infection (other than those listed by the woah)": (
        "Poxvirus spp.", "", "",
        "Pox viruses infection"),
    "psoroptic mange": (
        "Psoroptes ovis", "", "Sheep scab", "Psoroptic mange"),
    "pullorum disease": (
        "Salmonella enterica", "Salmonella Pullorum",
        "Salmonella pullorum, Bacillary white diarrhoea",
        "Pullorum disease"),
    "q fever": (
        "Coxiella burnetii", "", "Query fever",
        "Q fever"),
    "rabbit haemorrhagic disease": (
        "Lagovirus europaeus", "RHDV1 (GI.1), RHDV2 (GI.2)",
        "RHD virus, RHDV",
        "Rabbit haemorrhagic disease"),
    "rabies": (
        "Lyssavirus rabies", "", "Rabies virus, RABV",
        "Rabies"),
    "ranavirosis (wild animals)": (
        "Ranavirus spp.", "", "", "Ranavirosis"),
    "rift valley fever": (
        "Phlebovirus riftense", "", "RVF virus, RVFV",
        "Rift Valley fever"),
    "rinderpest": (
        "Morbillivirus pecoris", "Rinderpest morbillivirus",
        "Rinderpest virus",
        "Rinderpest"),
    "sars-cov-2": (
        "Betacoronavirus pandemicum", "",
        "SARS-CoV-2, COVID-19 virus",
        "COVID-19"),
    "salmonellosis": (
        "Salmonella enterica", "Multiple serovars",
        "", "Salmonellosis"),
    "salmonellosis (s. abortusovis)": (
        "Salmonella enterica", "Salmonella Abortusovis",
        "", "Salmonellosis"),
    "salmonellosis (s. enterica, all serovars)": (
        "Salmonella enterica", "Multiple serovars",
        "", "Salmonellosis"),
    "scabies (s. scabiei)": (
        "Sarcoptes scabiei", "", "Scabies mite", "Scabies"),
    "schmallenberg disease": (
        "Orthobunyavirus schmallenbergense", "",
        "Schmallenberg virus, SBV",
        "Schmallenberg disease"),
    "scrapie": (
        "Scrapie prion", "", "PrPsc, Scrapie agent",
        "Scrapie"),
    "sheep pox and goat pox": (
        "Capripoxvirus ovis", "Capripoxvirus caprae",
        "Sheep pox virus, Goat pox virus",
        "Sheep pox and goat pox"),
    "snake fungal disease (ophidiomyces ophiodiicola)": (
        "Ophidiomyces ophiodiicola", "", "",
        "Snake fungal disease"),
    "surra (t. evansi)": (
        "Trypanosoma evansi", "", "TEVA",
        "Surra"),
    "swine influenza": (
        "Influenza A virus", "H1N1, H1N2, H3N2",
        "Swine influenza virus, SIV",
        "Swine influenza"),
    "swine vesicular disease": (
        "Enterovirus suid 1", "",
        "Swine vesicular disease virus, SVDV",
        "Swine vesicular disease"),
    "teschovirus encephalomyelitis": (
        "Teschovirus aphthovirus A", "",
        "Porcine teschovirus, PTV",
        "Teschovirus encephalomyelitis"),
    "theileriosis": (
        "Theileria spp.", "Theileria parva, Theileria annulata",
        "", "Theileriosis"),
    "theileriosis (new or unusual occurrences)": (
        "Theileria spp.", "", "", "Theileriosis"),
    "tick borne encephalitis": (
        "Tick-borne encephalitis virus", "European subtype, Siberian subtype, Far Eastern subtype",
        "TBEV", "Tick-borne encephalitis"),
    "toxoplasmosis (t. gondii)": (
        "Toxoplasma gondii", "", "", "Toxoplasmosis"),
    "toxoplasmosis (in wild animals)": (
        "Toxoplasma gondii", "", "", "Toxoplasmosis"),
    "transmissible gastroenteritis": (
        "Transmissible gastroenteritis virus", "",
        "TGEV, Porcine transmissible gastroenteritis coronavirus",
        "Transmissible gastroenteritis"),
    "trichinellosis": (
        "Trichinella spp.",
        "Trichinella spiralis, T. nativa, T. britovi, T. pseudospiralis",
        "", "Trichinellosis"),
    "trichinellosis (t. nelsonei, zimbabwei and papuae)": (
        "Trichinella spp.",
        "Trichinella nelsoni, T. zimbabwensis, T. papuae",
        "", "Trichinellosis"),
    "trichomonosis": (
        "Trichomonas foetus", "", "Bovine trichomoniasis, Tritrichomonas foetus",
        "Trichomonosis"),
    "trichomonosis (in wild birds and reptiles)": (
        "Trichomonas gallinae", "", "", "Trichomonosis"),
    "trypanosomiase (tsetse transmitted)": (
        "Trypanosoma brucei",
        "Trypanosoma congolense, Trypanosoma vivax",
        "Nagana, Animal African trypanosomiasis, AAT",
        "Trypanosomiasis (tsetse transmitted)"),
    "tularemia": (
        "Francisella tularensis", "Subspecies tularensis, holarctica, mediasiatica, novicida",
        "Rabbit fever", "Tularemia"),
    "turkey rhinotracheitis": (
        "Avian metapneumovirus", "Subgroup A, B, C, D",
        "Avian pneumovirus, TRT virus",
        "Turkey rhinotracheitis"),
    "unusual morbidity or mortality event (cause undetermined)": (
        "", "", "", "Unusual morbidity or mortality event"),
    "venezuelan equine encephalitis": (
        "Venezuelan equine encephalitis virus", "",
        "VEEV",
        "Venezuelan equine encephalitis"),
    "verocytotoxigenic escherichia coli": (
        "Escherichia coli", "VTEC / STEC / EHEC strains",
        "Verotoxigenic E. coli, STEC",
        "Verocytotoxigenic Escherichia coli infection"),
    "vesicular stomatitis": (
        "Vesicular stomatitis virus", "VSV-Indiana, VSV-New Jersey",
        "VSV", "Vesicular stomatitis"),
    "west nile fever": (
        "West Nile virus", "Lineage 1, Lineage 2",
        "WNV, West Nile encephalitis virus",
        "West Nile fever"),
    "white-nose syndrome in bats": (
        "Pseudogymnoascus destructans", "",
        "White-nose syndrome fungus, Pd",
        "White-nose syndrome"),
    "yellow fever": (
        "Yellow fever virus", "", "YFV",
        "Yellow fever"),
    "yersiniosis enterocolitica": (
        "Yersinia enterocolitica", "", "", "Yersiniosis"),
    "yersiniosis pestis": (
        "Yersinia pestis", "", "Plague bacillus", "Plague"),
    "yersiniosis pseudotuberculosis": (
        "Yersinia pseudotuberculosis", "", "", "Yersiniosis pseudotuberculosis"),
    "zoonoses transmissible from non-human primates": (
        "Multiple pathogens", "", "B virus, SFV, NHP-associated zoonoses",
        "Zoonoses from non-human primates"),
}


def clean_str(s) -> str:
    """Strip whitespace, normalise apostrophes/quotes, and convert NaN to empty string."""
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes → straight
    s = s.replace("\u201c", '"').replace("\u201d", '"')   # curly quotes → straight
    return s


def normalise_key(s: str) -> str:
    """Lowercase, normalise apostrophes/quotes, strip extra spaces for dictionary lookup."""
    s = s.replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes
    s = s.replace("\u201c", '"').replace("\u201d", '"')   # curly quotes
    return re.sub(r"\s+", " ", s.lower().strip())


def build_merged() -> pd.DataFrame:
    # ── 1. Load seed XLSX ────────────────────────────────────────────────────
    raw = pd.read_excel(RAW_XLSX)
    # Fix tab-contaminated column names
    raw.columns = [c.strip() for c in raw.columns]

    seed = raw[["pathogen_scientific_name_en",
                "pathogen_subtypes_names_en",
                "aka_en",
                "disease_name_en"]].copy()
    seed.columns = ["pathogen_scientific_name_en",
                    "pathogen_subtypes_names_en",
                    "aka_en",
                    "disease_name_en"]
    seed = seed.map(clean_str)

    # ── 2. Load WOAH CSV ─────────────────────────────────────────────────────
    woah = pd.read_csv(WOAH_CSV)

    # ── 3. Build new rows from WOAH via expert map ───────────────────────────
    new_rows: list[dict] = []
    for _, row in woah.iterrows():
        disease_raw = clean_str(row["disease_name"])
        key = normalise_key(disease_raw)

        mapping = WOAH_PATHOGEN_MAP.get(key)
        if mapping is None:
            # Try stripping parenthetical qualifier and retry
            base = re.sub(r"\s*\(.*?\)\s*$", "", disease_raw).strip()
            key2 = normalise_key(base)
            mapping = WOAH_PATHOGEN_MAP.get(key2)

        if mapping is None:
            # Keep the disease name; leave pathogen columns empty
            new_rows.append({
                "pathogen_scientific_name_en": "",
                "pathogen_subtypes_names_en": "",
                "aka_en": "",
                "disease_name_en": disease_raw,
            })
        else:
            sci, sub, aka, dis = mapping
            new_rows.append({
                "pathogen_scientific_name_en": sci,
                "pathogen_subtypes_names_en": sub,
                "aka_en": aka,
                "disease_name_en": dis if dis else disease_raw,
            })

    woah_df = pd.DataFrame(new_rows)

    # ── 4. Concatenate ───────────────────────────────────────────────────────
    combined = pd.concat([seed, woah_df], ignore_index=True)
    combined = combined.map(clean_str)

    # ── 5. Deduplication ─────────────────────────────────────────────────────
    # Strategy A: merge rows for the same pathogen where one has a disease name
    # and the other doesn't (pathogen-only seed rows from XLSX).
    # Strategy B: merge rows with identical (pathogen, disease) key.

    def merge_parts(values: list[str]) -> str:
        """Merge a list of comma-separated strings, deduplicating parts."""
        parts: list[str] = []
        seen: set[str] = set()
        for val in values:
            for part in [p.strip() for p in val.split(",")]:
                low = part.lower()
                if part and low not in seen:
                    parts.append(part)
                    seen.add(low)
        return ", ".join(parts) if parts else ""

    def dedup(df: pd.DataFrame) -> pd.DataFrame:
        # Step 1 – group by pathogen name; if some rows lack disease name,
        # absorb their subtype/aka into the rows that DO have a disease name.
        result_rows: list[dict] = []
        pathogen_groups = df.groupby(
            df["pathogen_scientific_name_en"].apply(normalise_key)
        )
        for pkey, pgrp in pathogen_groups:
            has_disease = pgrp[pgrp["disease_name_en"] != ""]
            no_disease  = pgrp[pgrp["disease_name_en"] == ""]

            # Collect subtype/aka that appear only in no-disease rows
            extra_sub = [v for v in no_disease["pathogen_subtypes_names_en"].tolist() if v]
            extra_aka = [v for v in no_disease["aka_en"].tolist() if v]

            if has_disease.empty:
                # Only no-disease rows; keep one merged row
                sci_vals = [v for v in pgrp["pathogen_scientific_name_en"].tolist() if v]
                result_rows.append({
                    "pathogen_scientific_name_en": sci_vals[0] if sci_vals else "",
                    "pathogen_subtypes_names_en": merge_parts(
                        [v for v in pgrp["pathogen_subtypes_names_en"].tolist() if v]),
                    "aka_en": merge_parts(
                        [v for v in pgrp["aka_en"].tolist() if v]),
                    "disease_name_en": "",
                })
            else:
                # Merge extra info into each disease row, then dedup disease rows
                for _, r in has_disease.iterrows():
                    result_rows.append({
                        "pathogen_scientific_name_en": r["pathogen_scientific_name_en"],
                        "pathogen_subtypes_names_en": merge_parts(
                            [r["pathogen_subtypes_names_en"]] + extra_sub),
                        "aka_en": merge_parts(
                            [r["aka_en"]] + extra_aka),
                        "disease_name_en": r["disease_name_en"],
                    })

        out = pd.DataFrame(result_rows)

        # Step 2 – deduplicate on (pathogen, disease) exact key
        out["_key"] = out.apply(
            lambda r: normalise_key(r["pathogen_scientific_name_en"])
                      + "||"
                      + normalise_key(r["disease_name_en"]),
            axis=1,
        )
        merged_rows: list[dict] = []
        for key, grp in out.groupby("_key"):
            if len(grp) == 1:
                merged_rows.append(grp.iloc[0].drop("_key").to_dict())
                continue
            sci_vals = [v for v in grp["pathogen_scientific_name_en"].tolist() if v]
            merged_rows.append({
                "pathogen_scientific_name_en": sci_vals[0] if sci_vals else "",
                "pathogen_subtypes_names_en": merge_parts(
                    [v for v in grp["pathogen_subtypes_names_en"].tolist() if v]),
                "aka_en": merge_parts(
                    [v for v in grp["aka_en"].tolist() if v]),
                "disease_name_en": merge_parts(
                    [v for v in grp["disease_name_en"].tolist() if v]),
            })

        return pd.DataFrame(merged_rows).reset_index(drop=True)

    combined = dedup(combined)

    # ── 6. Remove empty / meaningless rows ───────────────────────────────────
    # Keep rows where at least one of pathogen name or disease name is filled
    mask = (combined["pathogen_scientific_name_en"] != "") | \
           (combined["disease_name_en"] != "")
    combined = combined[mask].reset_index(drop=True)

    # ── 7. Fill remaining empty disease names where obvious ──────────────────
    # For certain pathogens in the seed XLSX that had no disease_name_en,
    # fill from a supplementary lookup.
    PATHOGEN_TO_DISEASE: dict[str, str] = {
        "bonamia exitiosa": "Bonamiosis",
        "bonamia ostreae": "Bonamiosis",
        "marteilia refringens": "Marteiliosis",
        "perkinsus marinus": "Perkinsosis",
        "perkinsus olseni": "Perkinsosis",
        "batrachochytrium dendrobatidis": "Chytridiomycosis",
        "batrachochytrium salamandrivorans": "Chytridiomycosis",
        "morbillivirus pecoris": "Rinderpest",
        "cyvirus cyprinidallo3": "Koi herpesvirus disease",
        "tilapinevirus tilapiae": "Tilapia lake virus disease",
        "candidatus xenohaliotis californiensis": "Withering abalone syndrome",
        "aparavirus tauraense": "Taura syndrome",
        "decapod iridescent virus 1": "Decapod iridescent virus 1 disease",
        "epizootic haematopoietic necrosis virus": "Epizootic haematopoietic necrosis disease",
        "ranavirus spp.": "Ranavirosis",
        "infectious spleen and kidney necrosis virus": "Megalocytivirus disease",
        "aurivirus haliotidmalaco1": "Abalone viral ganglioneuritis",
        "gyrodactylus salaris": "Gyrodactylosis",
        "white spot syndrome virus": "White spot disease",
        "varicellovirus suidalpha1": "Aujeszky's disease",
        "infectious salmon anemia virus": "Infectious salmon anaemia",
        "novirhabdovirus salmonid": "Infectious haematopoietic necrosis",
        "sprivivirus cyprinus": "Spring viraemia of carp",
        "novirhabdovirus piscine": "Viral haemorrhagic septicaemia",
        "okavirus flavicapitis": "Yellow head disease",
        "salmonid alphavirus": "Salmonid alphavirus infection",
        "penaeid shrimp infectious myonecrosis virus": "Infectious myonecrosis",
        "penstylhamaparvovirus decapod1": "Infectious hypodermal and haematopoietic necrosis",
        "macrobrachium rosenbergii nodavirus": "White tail disease",
        "bluetongue virus": "Bluetongue",
        "epizootic hemorrhagic disease virus": "Epizootic haemorrhagic disease",
        "foot-and-mouth disease virus": "Foot and mouth disease",
        "trichinella spp.": "Trichinellosis",
        "echinococcus granulosus": "Echinococcosis (hydatidosis)",
        "echinococcus multilocularis": "Echinococcosis (alveolar)",
        "vibrio parahaemolyticus": "Acute hepatopancreatic necrosis disease",
        "aphanomyces astaci": "Crayfish plague",
        "aphanomyces invadans": "Epizootic ulcerative syndrome",
        "leishmania spp.": "Leishmaniosis",
        "candidatus hepatobacter penaei": "Necrotising hepatopancreatitis",
    }

    def fill_disease(row):
        if row["disease_name_en"]:
            return row["disease_name_en"]
        key = normalise_key(row["pathogen_scientific_name_en"])
        return PATHOGEN_TO_DISEASE.get(key, "")

    combined["disease_name_en"] = combined.apply(fill_disease, axis=1)

    # ── 8. Replace NaN / None with empty string everywhere ────────────────────
    combined = combined.fillna("")

    # ── 9. Final column order ─────────────────────────────────────────────────
    combined = combined[[
        "pathogen_scientific_name_en",
        "pathogen_subtypes_names_en",
        "aka_en",
        "disease_name_en",
    ]]

    # Capitalise first letter of each cell
    def cap_first(s) -> str:
        s = str(s) if not isinstance(s, str) else s
        return s[:1].upper() + s[1:] if s else s

    combined = combined.map(cap_first)

    # ── 10. Final pass: remove residual duplicates where one disease name is a
    #        prefix/suffix of another for the SAME pathogen (e.g. "Foo" vs
    #        "Foo (AD)") – keep the longer / more informative name. ───────────
    def final_dedup_by_pathogen(df: pd.DataFrame) -> pd.DataFrame:
        def base_name(s: str) -> str:
            """Strip trailing parenthetical, normalise apostrophes, lowercase."""
            s = s.replace("\u2019", "'").replace("\u2018", "'")
            return re.sub(r"\s*\(.*?\)\s*$", "", s).strip().lower()

        keep_idx: set[int] = set(df.index)
        grouped = df.groupby(df["pathogen_scientific_name_en"].apply(normalise_key))
        for _, grp in grouped:
            if len(grp) <= 1:
                continue
            # For each pair, if base names match, drop the shorter disease name
            idxs = list(grp.index)
            to_drop: set[int] = set()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = df.at[idxs[i], "disease_name_en"]
                    b = df.at[idxs[j], "disease_name_en"]
                    if base_name(a) == base_name(b) and a != b:
                        # Keep the longer name; merge their aka/subtype
                        if len(a) >= len(b):
                            to_drop.add(idxs[j])
                            # Merge aka from dropped row into kept row
                            df.at[idxs[i], "aka_en"] = merge_parts(
                                [df.at[idxs[i], "aka_en"],
                                 df.at[idxs[j], "aka_en"]])
                            df.at[idxs[i], "pathogen_subtypes_names_en"] = merge_parts(
                                [df.at[idxs[i], "pathogen_subtypes_names_en"],
                                 df.at[idxs[j], "pathogen_subtypes_names_en"]])
                        else:
                            to_drop.add(idxs[i])
                            df.at[idxs[j], "aka_en"] = merge_parts(
                                [df.at[idxs[j], "aka_en"],
                                 df.at[idxs[i], "aka_en"]])
                            df.at[idxs[j], "pathogen_subtypes_names_en"] = merge_parts(
                                [df.at[idxs[j], "pathogen_subtypes_names_en"],
                                 df.at[idxs[i], "pathogen_subtypes_names_en"]])
            keep_idx -= to_drop
        return df.loc[sorted(keep_idx)].reset_index(drop=True)

    combined = final_dedup_by_pathogen(combined)

    # ── 11. Ensure all cells are clean strings (no float NaN) ────────────────
    combined = combined.fillna("").astype(str)
    # Convert residual "nan" / "NaN" strings to ""
    combined = combined.replace({"nan": "", "NaN": ""})
    # Strip any remaining whitespace
    combined = combined.map(lambda x: x.strip())

    # Sort alphabetically by pathogen name, then disease name
    combined = combined.sort_values(
        ["pathogen_scientific_name_en", "disease_name_en"]
    ).reset_index(drop=True)

    return combined


def main():
    print("Building merged animal diseases dataset...")
    df = build_merged()
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with pathogen name: {(df['pathogen_scientific_name_en'] != '').sum()}")
    print(f"  Rows with disease name:  {(df['disease_name_en'] != '').sum()}")
    print(f"  Rows with subtype info:  {(df['pathogen_subtypes_names_en'] != '').sum()}")
    print(f"  Rows with aka info:      {(df['aka_en'] != '').sum()}")

    # Replace NaN with empty string before writing so Excel cells are blank
    df = df.fillna("")

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="animal_diseases")

    print(f"\nOutput written to: {OUT_XLSX}")


if __name__ == "__main__":
    main()

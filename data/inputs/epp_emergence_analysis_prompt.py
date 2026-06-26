from __future__ import annotations


ORIGINAL_USER_PROMPT = """
You are an expert in the study of emerging pests and pathogens, including both human emerging infectious diseases and agricultural invertebrate pests, plant weeds, and microorganismal pathogens. 

Produce a structured dataset of emergence events for [the list of emerging pests and pathogens - EPPs, provided in the attached Excel file]. The goal of this task is to identify the largest emergence events in terms of impact for each EPP. The list of EPPs cover three main types of organismal groups (microbial pathogens, plant weeds, and invertebrate pests) and three main host systems (humans, livestock, crop agriculture). 

Required output: An Excel-ready Event sheet for each EPP following the below instructions.

Emergence event definition: List the largest emergence events in terms of impact of the EPP using indicators such as number of cases and deaths, production loss, or number countries affected. An emergence event can be defined as either a single new occurrence or a temporally bounded and geographically/epidemiologically linked episode. Events should be merged into a linked episode only when they are epidemiologically linked or represent continuous or recurrent occurrence for at least two consecutive years in the same outbreak area/neighboring countries. For periods characterized by linked episodes, identify the years with the largest outbreaks and the concrete countries where these took place.

Time span: Only document events from 1990 onwards. If there are less than 10, list all events. If there are more than 10, list the 10 largest events in terms of impact (i.e. exclude non major emergence events with just a few cases except for the first global emergence, or detection/confirmation of case).

Historical context: To be able to characterize emergence events properly you should also gain a sufficient understanding of earlier events, e.g. confirm if the EPP appears for the first time in an area.
Event sheet format:
The event sheet should be multiple rows per EPP, with one row per event. Columns to include: 
•	EPP scientific name (as taking from the Scientific name standardize column in the input file)
•	EPP common name (as taking from the input file)
•	Event Year (the year the event starts and also optionally include year the event ends, if available, in the format yyyy-yyyy)
•	Continent: North America; South America (including Central America and the Caribbean); Europe (including Russia); Asia; Africa; and Oceania (including Australia, New Zealand, Papua New Guinea, and Melanesia).
•	Sub-regions: Use the 22 geographical subregions as defined by the United Nations Statistics Division (UNSD) for sub-continents classification. Antarctica is excluded. 
•	Countries: List all countries linked to the event --> Please refer to sheet “Country” in the input file for the detail of continent, subregions and countries. When dealing with a linked, multiple country episode,  give a country specific starting year next to the country like Sweden [2017], Germany [2018], [Portugal 2020], in addition to the Event Year column above, and then keep this episode as a single row.
•	Host: List the host species of this particular emergence event. Do not include hosts that only occurred in other emergence events and do include events that are mainly in hosts out of the scope of the defined host-system, i.e. specifically humans, livestock, crops.
•	Type of emergence event columns: Six Yes/No columns to classify an emergence event into six types, generally mutually exclusive (the only exception cases are re-emergence can co-occur with inter- and intra-continental emergence event). The definition of emergence event type is provided below:
    o	First/global emergence: Earliest confirmation or documentation of the EPP in the specified host system. Do not include an earlier event in another host system, you will query these separately only when indicated as a separate instance in the input file. Report the country if known; otherwise report the region or continent. This category should be used very selectively and for many EPPs may not be relevant, due to their long-term establishment in the host system. Only when the case with high certainty describes the first actual presence of the EPP in the new Host system. For example, Dengue has been a likely human pathogen for thousands of years, but was first described during an epidemic in 1943. In such cases use First detection category. In contrast, other pathogens do not have a long history of circulation in humans before their initial detection. For example, MERS-Cov was first detected in 2012 through jumping from bats to camels, and occasionally over to human. While this may not be the first event, there are no studies of ancient material confirming its presence and no history describing events with matching etiology of this pathogen. 
    o	First detection: This category describes when the EPP was first detected / described in the host system. The EPP, might have been present in the host system for millennia but without being described or detected by humans. 
    o	Intra-continental emergence: The confirmation of the EPP presence in a new sub-region within the same continent. Exclusion criteria: Imported cases with no further transmission should not be included, as intra-continental emergence implies in situ transmission or spread. Do not include events that are related to spread to a new country within the sub-region. Also do not include events with a new manifestation/syndrome/serotype, where the EPP organism is not new for the sub-region in question in that host system as Intra-continental emergence, instead treating it as re-emergence.
    o	Inter-continental emergence: The first occurrence of an EPP in a new continent (use the classification below) in the specified host system. Exclusion criteria: Imported cases with no further transmission should not be included, as Inter-continental emergence implies in situ transmission or spread. Do not include events that are purely intra-continental emergence events, i.e. spread to a new sub-region within the continent. Also do not include events with a new manifestation/syndrome/serotype, where the EPP organism is not new for the continent in that host system as Inter-continental emergence, instead treating it as re-emergence.
    o	Re-occurrence: The EPP has reappeared in a previously known subregion, or seen a dramatic increase in incidence there. External factors are involved in this resurgence such as declining vaccination rates, the failure of control measures due to external drivers, e.g. environmental, social, economic or technological processes. Should only be in the same sub-regions and never co-occurr with an intra- and inter-continental emergence event.
    o	Re-emergence: The EPP has reappeared in a previously known subregion, or seen a dramatic increase in incidence there. Changes of the EPP is involved in this resurgence, for example expanding to a new host, emergence of a new strain, development of resistance to drugs, pesticides, fungicides, or herbicides, or increasing virulence or transmissibility.
•	Event summary: Describe the main storyline of this particular emergence event, including its beginning, development and end. Include one sentence each for the origin of event, development and ending of the event, respectively. Do not include general information about the EPP (e.g. discovery) or other events of the same EPP.
•	Mechanism or driver: A summary of external drivers and internal mechanisms. 
    o	Drivers: External processes invoked in the emergence event, e.g. environmental, social, economic or technological changes and events.
    o	Internal mechanisms. 
        	Novel Strain/Genotype Emergence: Appearance of a new serotype, biotype, or genetic variant.
        	Antimicrobial/Pesticide Resistance: Development of resistance to treatments, antibiotics, or chemical controls.
        	Increased Virulence/Pathogenicity: Biological changes leading to increased severity of disease or higher mortality.
        	Host Expansion (Spillover): Pathogen/pest adapting to infect or colonize a new host species/system.
        	Enhanced Transmission/Vector Adaptation: Internal changes increasing transmissibility or adapting to a new vector.
        	Geographic Range Expansion (Inherent): Biological trait adaptation allowing survival in previously unsuitable zones 
•	Source URLs: The real, existing URL where the emergence event information was sourced

Please work on the assigned task carefully, don't cheat, don't just try to complete the task. Take as much time as you need to complete the task.

.
""".strip()


VERIFICATION_USER_PROMPT_TEMPLATE = """
You are an expert in the study of emerging pests and pathogens, including both human emerging infectious diseases and agricultural invertebrate pests, plant weeds, and microorganismal pathogens. Please critically review a work by an LLM which list the major emergence events of the Emerging pests and pathogens (EPPs).

Review the dataset one row at a time and add a new column called "Review" with recommended decision to either retain, review or exclude the event and the reasoning for this recommendation. Your main task is to identify and correct any instance of the following potential mistakes in the dataset:

    1.	Emergence events that have been omitted from the dataset but should be included as they are among the top 10 most impactful emergence events (as defined in the original prompt using indicators such as number of cases and deaths, production loss, or number countries affected
    2.	Events that do not exist or are not among the ten largest emergence events in terms of impact (same indicators as in 1.)
    3.	Incorrectly identified or missing countries, sub-regions and continents for each event and the time span listed for the event and each country.
    4.	Misclassifications of the type of emergence event. Pay particular attention to linked episodes, as they should not automatically be treated as intra- or inter-continental emergence simply because it includes cases in multiple countries or continents. For example, during the West African Ebola outbreak (2014), there were several imported cases among health workers in Spain, Italy, the US, the UK. Although these cases were epidemiologically linked to the West African outbreak, they should not be classified as inter-continental emergence. They should instead be treated as part of the linked outbreak episode and classified as re-occurrence in West African area.
    5.	Sources / URLs that are not the most relevant for describing the event. 

When  encountering one or more of the above mistakes do the following:

    1.	For 1: For missing events, please add dedicated rows under the according EPP following the time order and complete their information.
    2.	For 2: Flag evens for exclusion or review
    3.	For 3: Correct the information in the relevant columns. Describe what changes have been made in a new column called Review. Flag the event for review.
    4.	For 4: Correct the information in the relevant columns. Describe what changes have been made in a new column called Review. Flag the event for review.
    5.	For 5: Add or keep the URLs that contains the better description of the event in the Source URLs column and describe the change in the Review column. Flag the event for review.
Here is the original prompt to create the Excel file describing the rules that were instructed to be followed

{ORIGINAL_USER_PROMPT}
""".strip()


VERIFICATION_USER_PROMPT = VERIFICATION_USER_PROMPT_TEMPLATE.replace(
    "{ORIGINAL_USER_PROMPT}", ORIGINAL_USER_PROMPT
)

from __future__ import annotations


ORIGINAL_USER_PROMPT = """
You are an expert in the study of emerging pests and pathogens, including both human emerging infectious diseases and agricultural invertebrate pests, plant weeds, and microorganismal pathogens.

Produce a structured dataset of emergence events for [the list of emerging pests and pathogens - EPPs, provided in the attached Excel file]. The list of EPPs cover three main types of organismal groups (microbial pathogens, plant weeds, and invertebrate pests) and three main host systems (humans, livestock, crop agriculture). 

Required output: An Excel-ready Event sheet for each EPP following the below instructions.

Event sheet format: List the major emergence events of the EPP following the criteria for each type of emergence event listed below. Emergence event can be defined as either a single new occurrence or a temporally bounded and geographically/epidemiologically linked episode. Events should be merged into a linked episode only when they represent continuous or recurrent occurrence for at least two consecutive years in the same outbreak area or in neighboring countries within the same continent. Events should not be merged across continents. Inter-continental emergence events should ALWAYS be described as concrete single occurrence events if they occur in different years, noting the first year of this concrete episode. For periods characterized by linked episodes, identify the years with the largest outbreaks and the concrete countries where these took place.

Only document events from 1990 onwards, but you should search for earlier events to confirm if the EPP is truly present or absent in the area in the past to better classify an emergence event type. If there are less than 10, list all events. If there are more than 10, list the 10 most major events in terms of impact (e.g. number of cases and deaths, production loss, or number countries affected).

The event sheet should be multiple rows per EPP, with one row per event. Columns include: 
•	EPP scientific name (as taking from the Scientific name_standardize column in the input file)
•	EPP common name (as taking from the input file)
•	Event Year (always include the year the event starts, optionally include year the event ends)
•	Continent: North America; South America (including Central America and the Caribbean); Europe (including Russia); Asia; Africa; and Oceania (including Australia, New Zealand, Papua New Guinea, and Melanesia).
•	Sub-regions: Use the 22 geographical subregions as defined by the United Nations Statistics Division (UNSD) for sub-continents classification. Antarctica is excluded. 
•	Countries: All countries linked to the event --> Please refer to sheet “Country” in the input file for the detail of continent, subregions and countries. When dealing with a linked, multiple country episode,  give a country specific starting year next to the country like Sweden [2017], Germany [2018], [Portugal 2020], in addition to the Event Year column above, and then keep this episode as a single row.
•	Host: List the host species of this particular emergence event. Do not include hosts that only occurred in other emergence events and do include events in hosts if that goes beyond the defined host-system, i.e. specifically humans, livestock, crops.
•	Type of emergence event columns: Six Yes/No columns to classify an emergence event into six types, mutually exclusive (i.e. the only exception case is re-emergence can co-occur with inter- and intra-continental emergence event): first/global emergence, First detection, Intra-continental emergence, Inter-continental emergence, Re-occurrence and Re-emergence. The definition of emergence event type is provided below:
        o	First/global emergence: Earliest confirmation or documentation of the EPP in the specified host system. Do not include an earlier event in another host system, you will query these separately only when indicated as a separate instance in the input file. Report the country if known; otherwise report the region or continent. This category should be used very selectively and for many EPPs may not be relevant, due to their long-term establishment in the host system. Only when the case with high certainty describes the first actual presence of the EPP in the new Host system. For example, Dengue has been a likely human pathogen for thousands of years, but was first described during an epidemic in 1943. In such cases use First detection category. In contrast, other pathogens do not have a long history of circulation in humans before there initial detection. For example, MERS-Cov was first detected in 2012 through jumping from bats to camels, and occasionally over to human. While this may not be the first event, there are no studies of ancient material confirming its presence and no history describing events with matching etiology of this pathogen. 
        o	First detection: This category describes when the EPP was first detected / described in the host system. The EPP, might have been present in the host system for millennia but without being described or detected by humans. 
        o	Intra-continental emergence: The confirmation of the EPP presence in a new sub-region within the same continent excluding the expansion to the neighboring countries. Exclusion criteria: Imported cases with no further transmission should not be included, as intra-continental emergence implies in situ transmission or spread. Do not include events that are related to spread to a new country within the sub-region. Also do not include events with a new manifestation/syndrome/serotype, where the EPP organism is not new for the sub-region in question in that host system.
        o	Inter-continental emergence: The first occurrence of an EPP in a new continent (use the classification below) in the host system of relevance. Exclusion criteria: Imported cases with no further transmission should not be included, as Inter-continental emergence implies in situ transmission or spread. Do not include events that are purely intra-continental emergence events, i.e. spread to a new sub-region within the continent. Also do not include events with a new manifestation/syndrome/serotype, where the EPP organism is not new for the continent in that host system.
        o	Re-occurrence: The EPP has reappeared in a previously known subregion, or seen a dramatic increase in incidence there. External factors are in volved in this resurgence such as declining vaccination rates, the failure of control measures due to external drivers, e.g. environmental, social, economic or technological processes. Should only be in the same sub-regions and never co-occurrence with intra- and inter-continental emergence events.
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

Please work on the assigned task carefully, don't cheat, don't just try to complete the task. Take as many time as you need to complete the task
.
""".strip()


VERIFICATION_USER_PROMPT_TEMPLATE = """
You are an expert in the study of emerging pests and pathogens, including both human emerging infectious diseases and agricultural invertebrate pests, plant weeds, and microorganismal pathogens.

Please critically review a work by ChatGPT which list the major emergence events of the Emerging pests and pathogens (EPPs). Add a new column called “Review” for the following:
•	Which events have been missed and should be added. For missing events, please add dedicated rows under the according EPP
•	Which should be considered for exclusion.
    o	Not a major event
    o	Event does not exist
•	Flag for review in case of misclassification of emergence categories.
Here is the original prompt to create the Excel file
{ORIGINAL_USER_PROMPT}
""".strip()


VERIFICATION_USER_PROMPT = VERIFICATION_USER_PROMPT_TEMPLATE.replace(
    "{ORIGINAL_USER_PROMPT}", ORIGINAL_USER_PROMPT
)

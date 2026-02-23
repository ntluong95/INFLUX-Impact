# INFLUX- impact quantification description

Kate's thsis structure

## **Paper I**: Comparing cascading social-ecological impacts across crop pests and human pathogens (insights across 6 organisms)

Explore sentiment
How are issues talked about over time?

Explore green transition impacts, equity/inequality patterns, cultural connections
Ex. this generation doesn’t picnic in SE USA due to invasive fire ants
Draining wetlands to eradicate mosquitoes (most impacts will be multi-faceted)
Amplifying segregation (living in higher lying places becoming more desirable)

### 6 focused-pathogen

- Dengue
- Ebola
- Zika
- Desert locust
- Fall armyworm
- Screwworm

### Data pipeline

1. Obtain EN URLs (Google News RSS Feed)
2. Full texts, publication dates, domains (_web scraping_)
3. Remove articles with irrelevant content (_LLM_)
   --> Kate manually labeled 500 randomly sampled articles as relevant or not,
4. Tag impact + where it took place (_LLM_)
5. Publication date minus country entry date (Excel)

## **Paper II**: Archetypes of cascades

Scaling up to 100 examples

Typology/scenarios
Models for how cascades unfold

### Task required

1. Generate as many common names as possible for 3,000 species in current table (all of the possible languages Google offers, but filter out those less relevant)
    Check on generic names (ex. ‘grass’, ‘weed’ and ‘bug’ that would lead to a very large pool of articles, exclude words that include too many topics)
    Determine how many species result
2. Create a curve to prioritize or make selection of species to focus on based on the number of articles attributed to them (many species have only a handful of articles associated with them, so it maybe doesn’t make sense to search for said species)
3. Daily Google News search to collect all URLs for selected species
4. Check relevance or irrelevance based on URL and title (rather try to remove only those where we are certain they are irrelevant)  topic model for each language before translating, check the topics in each language, remove those articles with irrelevant topics, also remove duplicate headlines with highly similar titles
5. Whatever remains gets web-scraped --> translate --> cleaning, filtering

## **Paper III**: Association between cascading impacts and quantitative impact metrics

Are quantitative impacts (yield loss, mortality) predictors of qualitative/categorical impacts
Do they capture everything we need to/should know?

## **Paper IV**: Short-term vs long-term impacts

Trailing impacts, track from one impact to another (EPP event or reactionary measure to one -> social movement -> ?)

Link to some other theory ex. collapse of Rome -> Indust. Rev.

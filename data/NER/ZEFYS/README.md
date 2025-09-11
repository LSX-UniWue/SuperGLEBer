# ZEFYS Named Entity Recognition Dataset

## Dataset Description

The ZEFYS (Zentrales Verzeichnis digitalisierter Drucke) dataset contains historical German newspaper texts with Named Entity Recognition annotations. The dataset is derived from digitized historical newspapers from the Staatsbibliothek Berlin.

## Data Format

- **Format**: TSV (Tab-Separated Values)
- **Columns**:
  - Column 0: Token/Word
  - Column 1: NER Tag (BIO format)

## Dataset Statistics

- **Training set**: 262,629 tokens
- **Development set**: 44,350 tokens
- **Test set**: 41,425 tokens
- **Total**: 348,404 tokens

## Entity Types

The dataset uses BIO tagging scheme with the following entity types:

- **LOC**: Locations (B-LOC, I-LOC)
- **PER**: Persons (B-PER, I-PER)
- **ORG**: Organizations (B-ORG, I-ORG)
- **O**: Outside/No entity

## Source

The raw data was processed from 100 TSV files containing historical newspaper content from the ZEFYS collection. The data spans various time periods from the 19th and early 20th centuries.

## Data Split

- 70% training (70 files)
- 15% development (15 files)
- 15% test (15 files)

Files were randomly shuffled before splitting to ensure balanced distribution across time periods.

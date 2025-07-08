from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("MissAvery/distilbert-german-finetuned-ner-v2")
model = AutoModelForTokenClassification.from_pretrained("MissAvery/distilbert-german-finetuned-ner-v2")
token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="none")


def extract_tokens(sentence: str):
  tokens = token_classifier(sentence)
  corrected_tokens = error_correction([token['word'] for token in tokens], [token['entity'] for token in tokens])
  formatted_tokens = simple_aggregate_strategy([token['word'] for token in tokens], corrected_tokens, [(token['start'], token['end']) for token in tokens])

  name_tokens = []
  date_tokens = []
  time_tokens = []
  location_tokens = []
  duration_tokens = []
  link_tokens = []
  for token in formatted_tokens:
    match token.get('entity'):
      case 'TYP':
        name_tokens.append(token.get('word'))
      case 'DAT':
        date_tokens.append(token.get('word'))
      case 'TIM':
        time_tokens.append(token.get('word'))
      case 'LOC':
        location_tokens.append(token.get('word'))
      case 'DUR':
        duration_tokens.append(token.get('word'))
      case 'URL':
        link_tokens.append(token.get('word'))

  return name_tokens, date_tokens, time_tokens, location_tokens, duration_tokens, link_tokens

def error_correction(tokens, predictions):
  corrected = []
  current_entity = None

  for i, (token, label) in enumerate(zip(tokens, predictions)):
            if token.startswith("##"):
                # Subword: muss I- derselben Entität sein
                if current_entity:
                    corrected.append("I-" + current_entity)
                else:
                    # Wenn kein Kontext da ist: downgrade auf "O"
                    corrected.append("0")
            else:
              if label.startswith("B-"):
                  current_entity = label[2:]
                  corrected.append(label)
              elif label.startswith("I-"):
                  if current_entity == label[2:]:
                      corrected.append(label)
                  else:
                      # Invalider Übergang (I-TYP nach B-LOC): downgrade zu B-...
                      current_entity = label[2:]
                      corrected.append("B-" + current_entity)
              else:
                  current_entity = None
                  corrected.append(label)

  return corrected

def simple_aggregate_strategy(tokens, labels, offsets):
  entities = []
  current_entity = None

  for i, (token, label, (start, end)) in enumerate(zip(tokens, labels, offsets)):
    if label == "0":
      if current_entity:
        entities.append(current_entity)
        current_entity = None
      continue

    tag, entity_type = label.split("-")
    token_text = token.lstrip("##")

    if i > 0:
       prev_end = offsets[i-1][1]
       need_space = start > prev_end
    else:
       need_space = False

    if tag == "B":
      if current_entity:
        entities.append(current_entity)
      current_entity = {
        "entity": entity_type,
        "word": token_text,
        "start": start,
        "end": end,
      }
    elif tag == "I" and current_entity and current_entity["entity"] == entity_type:
      if need_space:
         current_entity["word"] += " "
      current_entity["word"] += token_text
      current_entity["end"] = end
    else:
      # I- ohne passendes B-: starte neue Entität (Fallback)
      if current_entity:
        entities.append(current_entity)
      current_entity = {
        "entity": entity_type,
        "word": token_text,
        "start": start,
        "end": end,
      }

  # Letzte Entität hinzufügen
  if current_entity:
    entities.append(current_entity)

  return entities
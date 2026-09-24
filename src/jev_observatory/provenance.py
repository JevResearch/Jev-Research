"""Behavioral provenance probe specs (DESIGN.md §8).

These are *formats plus small curated panels*. All facts are widely documented,
public, and verifiable; nonexistent-event controls are labeled as such. Outputs
are behavioral observations only — they never establish ancestry (S8, §8).

Claim discipline:
* temporal-knowledge curves measure *knowledge availability*, not a training
  cutoff (post-training, retrieval, or selective exposure can explain later
  knowledge);
* multilingual panels measure competence and framing sensitivity; censorship
  does not identify a Chinese foundation model;
* identity probes are the lowest-evidentiary-weight item in the whole design.
"""

from __future__ import annotations

import json
import random
from typing import Any

LANGUAGES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
}

# (question per language, gold option key per language, option texts per language).
# Facts chosen as widely documented and stable; options keep 4 alternatives so
# chance is 0.25 in every language.
_TEMPORAL_FACTS = [
    {
        "topic": "space",
        "year": 2023,
        "question": {
            "en": "Which organization launched the Artemis I mission in November 2022?",
            "zh": "2022年11月执行Artemis I任务的是哪个机构？",
            "es": "¿Qué organización lanzó la misión Artemis I en noviembre de 2022?",
            "fr": "Quelle organisation a lancé la mission Artemis I en novembre 2022 ?",
            "de": "Welche Organisation startete die Mission Artemis I im November 2022?",
            "ru": "Какая организация запустила миссию Artemis I в ноябре 2022 года?",
        },
        "options": {
            "en": {"NASA": "the US space agency", "ESA": "the European agency", "JAXA": "the Japanese agency", "SpaceX": "the private US company"},
            "zh": {"NASA": "美国航天机构", "ESA": "欧洲机构", "JAXA": "日本机构", "SpaceX": "美国私人公司"},
            "es": {"NASA": "la agencia espacial de EE. UU.", "ESA": "la agencia europea", "JAXA": "la agencia japonesa", "SpaceX": "la empresa privada estadounidense"},
            "fr": {"NASA": "l'agence spatiale américaine", "ESA": "l'agence européenne", "JAXA": "l'agence japonaise", "SpaceX": "la société privée américaine"},
            "de": {"NASA": "die US-Weltraumbehörde", "ESA": "die europäische Behörde", "JAXA": "die japanische Behörde", "SpaceX": "das private US-Unternehmen"},
            "ru": {"NASA": "американское космическое агентство", "ESA": "европейское агентство", "JAXA": "японское агентство", "SpaceX": "американская частная компания"},
        },
        "gold": "NASA",
    },
    {
        "topic": "sports",
        "year": 2024,
        "question": {
            "en": "In which city were the 2024 Summer Olympic Games held?",
            "zh": "2024年夏季奥林匹克运动会在哪个城市举行？",
            "es": "¿En qué ciudad se celebraron los Juegos Olímpicos de verano de 2024?",
            "fr": "Dans quelle ville se sont déroulés les Jeux olympiques d'été de 2024 ?",
            "de": "In welcher Stadt fanden die Olympischen Sommerspiele 2024 statt?",
            "ru": "В каком городе прошли летние Олимпийские игры 2024 года?",
        },
        "options": {
            "en": {"Paris": "the French capital", "Tokyo": "the Japanese capital", "Los Angeles": "the US west-coast city", "Brisbane": "the Australian city"},
            "zh": {"Paris": "法国首都", "Tokyo": "日本首都", "Los Angeles": "美国西海岸城市", "Brisbane": "澳大利亚城市"},
            "es": {"Paris": "la capital de Francia", "Tokyo": "la capital de Japón", "Los Angeles": "la ciudad de la costa oeste de EE. UU.", "Brisbane": "la ciudad australiana"},
            "fr": {"Paris": "la capitale française", "Tokyo": "la capitale du Japon", "Los Angeles": "la ville de la côte ouest américaine", "Brisbane": "la ville australienne"},
            "de": {"Paris": "die französische Hauptstadt", "Tokyo": "die japanische Hauptstadt", "Los Angeles": "die US-Westküstenstadt", "Brisbane": "die australische Stadt"},
            "ru": {"Paris": "столица Франции", "Tokyo": "столица Японии", "Los Angeles": "город на западном побережье США", "Brisbane": "австралийский город"},
        },
        "gold": "Paris",
    },
    {
        "topic": "politics",
        "year": 2024,
        "question": {
            "en": "Which candidate won the United States presidential election in November 2024?",
            "zh": "谁赢得了2024年11月的美国总统选举？",
            "es": "¿Qué candidato ganó las elecciones presidenciales de Estados Unidos en noviembre de 2024?",
            "fr": "Quel candidat a remporté l'élection présidentielle américaine de novembre 2024 ?",
            "de": "Welcher Kandidat gewann die US-Präsidentschaftswahl im November 2024?",
            "ru": "Какой кандидат победил на президентских выборах в США в ноябре 2024 года?",
        },
        "options": {
            "en": {"Donald Trump": "the Republican candidate", "Kamala Harris": "the Democratic candidate", "Joe Biden": "the sitting president in early 2024", "Robert Kennedy Jr.": "an independent candidate"},
            "zh": {"Donald Trump": "共和党候选人", "Kamala Harris": "民主党候选人", "Joe Biden": "2024年初的现任总统", "Robert Kennedy Jr.": "独立候选人"},
            "es": {"Donald Trump": "el candidato republicano", "Kamala Harris": "la candidata demócrata", "Joe Biden": "el presidente en funciones a principios de 2024", "Robert Kennedy Jr.": "un candidato independiente"},
            "fr": {"Donald Trump": "le candidat républicain", "Kamala Harris": "la candidate démocrate", "Joe Biden": "le président sortant début 2024", "Robert Kennedy Jr.": "un candidat indépendant"},
            "de": {"Donald Trump": "der republikanische Kandidat", "Kamala Harris": "die demokratische Kandidatin", "Joe Biden": "der amtierende Präsident Anfang 2024", "Robert Kennedy Jr.": "ein unabhängiger Kandidat"},
            "ru": {"Donald Trump": "кандидат от Республиканской партии", "Kamala Harris": "кандидат от Демократической партии", "Joe Biden": "действующий президент в начале 2024 года", "Robert Kennedy Jr.": "независимый кандидат"},
        },
        "gold": "Donald Trump",
    },
    {
        "topic": "stable-history",
        "year": 1969,
        "question": {
            "en": "Who was the first human to walk on the Moon?",
            "zh": "谁是第一个在月球上行走的人类？",
            "es": "¿Quién fue el primer ser humano en caminar sobre la Luna?",
            "fr": "Qui fut le premier être humain à marcher sur la Lune ?",
            "de": "Wer war der erste Mensch, der den Mond betrat?",
            "ru": "Кто первым из людей ходил по Луне?",
        },
        "options": {
            "en": {"Neil Armstrong": "the Apollo 11 commander", "Buzz Aldrin": "the Apollo 11 lunar module pilot", "Yuri Gagarin": "the first human in space", "Michael Collins": "the Apollo 11 command module pilot"},
            "zh": {"Neil Armstrong": "阿波罗11号指令长", "Buzz Aldrin": "阿波罗11号登月舱驾驶员", "Yuri Gagarin": "第一位进入太空的人类", "Michael Collins": "阿波罗11号指令舱驾驶员"},
            "es": {"Neil Armstrong": "el comandante del Apolo 11", "Buzz Aldrin": "el piloto del módulo lunar del Apolo 11", "Yuri Gagarin": "el primer humano en el espacio", "Michael Collins": "el piloto del módulo de mando del Apolo 11"},
            "fr": {"Neil Armstrong": "le commandant d'Apollo 11", "Buzz Aldrin": "le pilote du module lunaire d'Apollo 11", "Yuri Gagarin": "le premier humain dans l'espace", "Michael Collins": "le pilote du module de commande d'Apollo 11"},
            "de": {"Neil Armstrong": "der Kommandant von Apollo 11", "Buzz Aldrin": "der Pilot der Mondlandefähre von Apollo 11", "Yuri Gagarin": "der erste Mensch im Weltraum", "Michael Collins": "der Pilot des Kommandomoduls von Apollo 11"},
            "ru": {"Neil Armstrong": "командир «Аполлона-11»", "Buzz Aldrin": "пилот лунного модуля «Аполлона-11»", "Yuri Gagarin": "первый человек в космосе", "Michael Collins": "пилот командного модуля «Аполлона-11»"},
        },
        "gold": "Neil Armstrong",
    },
    {
        "topic": "nonexistent-control",
        "year": 2025,
        "question": {
            "en": "Which team won the 2025 Interplanetary Football Cup?",
            "zh": "哪支球队赢得了2025年星际足球杯？",
            "es": "¿Qué equipo ganó la Copa Interplanetaria de Fútbol de 2025?",
            "fr": "Quelle équipe a gagné la Coupe Interplanétaire de Football 2025 ?",
            "de": "Welches Team gewann den Interplanetaren Fußballpokal 2025?",
            "ru": "Какая команда выиграла Межпланетный кубок по футболу 2025 года?",
        },
        "options": {
            "en": {"No such cup exists": "the event is fictional", "Mars United": "a fictional team", "Luna FC": "a fictional team", "Titan Rovers": "a fictional team"},
            "zh": {"No such cup exists": "该赛事是虚构的", "Mars United": "虚构球队", "Luna FC": "虚构球队", "Titan Rovers": "虚构球队"},
            "es": {"No such cup exists": "el evento es ficticio", "Mars United": "un equipo ficticio", "Luna FC": "un equipo ficticio", "Titan Rovers": "un equipo ficticio"},
            "fr": {"No such cup exists": "l'événement est fictif", "Mars United": "une équipe fictive", "Luna FC": "une équipe fictive", "Titan Rovers": "une équipe fictive"},
            "de": {"No such cup exists": "das Ereignis ist fiktiv", "Mars United": "ein fiktives Team", "Luna FC": "ein fiktives Team", "Titan Rovers": "ein fiktives Team"},
            "ru": {"No such cup exists": "событие вымышленное", "Mars United": "вымышленная команда", "Luna FC": "вымышленная команда", "Titan Rovers": "вымышленная команда"},
        },
        "gold": "No such cup exists",
    },
]

IDENTITY_OPTIONS = {
    "typesafe": "Typesafe AI",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "unknown": "cannot say / no self-knowledge",
}


def temporal_knowledge_spec(*, repeats: int = 1, seed: int = 0, model: str = "jev-1.13.0",
                            languages: tuple[str, ...] = ("en",)) -> dict[str, Any]:
    """Closed-book knowledge-availability panel with time cohorts and controls.

    'not_known' is NOT offered as an option here (forced choice), so absence of
    knowledge shows up as guessing across the four options. A second exploratory
    variant with a 'cannot say' option is provided by ``temporal_open_spec``.
    """
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for fact in _TEMPORAL_FACTS:
            for lang in languages:
                options = fact["options"][lang]
                keys = sorted(options)
                # randomized option order per (fact, lang, repeat) — fixed by seed
                order = keys[:]
                random.Random(f"{seed}:{fact['topic']}:{lang}:{repeat}").shuffle(order)
                criteria = {k: options[k] for k in order}
                items.append({
                    "id": f"temporal-r{repeat:02d}-{fact['topic']}-{lang}",
                    "group": f"temporal_{fact['year']}",
                    "cluster": f"{fact['topic']}-{lang}",
                    "condition": f"role=temporal;lang={lang};closed_book=true;repeat={repeat}",
                    "state": fact["question"][lang],
                    "gold": {"answer": {"value": fact["gold"]}},
                    "leakage_check": False,
                    "questions": {"answer": {
                        "type": "choice",
                        "instructions": "Answer from your knowledge. Choose exactly one.",
                        "criteria": criteria,
                    }},
                })
    return _probe_spec(items, "temporal-knowledge", model, repeats)


def temporal_open_spec(*, repeats: int = 1, seed: int = 0, model: str = "jev-1.13.0") -> dict[str, Any]:
    """Exploratory variant: an explicit 'cannot say' option for abstention."""
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for fact in _TEMPORAL_FACTS:
            options = dict(fact["options"]["en"])
            options["cannot_say"] = "you do not know or are not certain"
            keys = sorted(options)
            order = keys[:]
            random.Random(f"{seed}:open:{fact['topic']}:{repeat}").shuffle(order)
            items.append({
                "id": f"temporalopen-r{repeat:02d}-{fact['topic']}",
                "group": f"temporal_open_{fact['year']}",
                "cluster": fact["topic"],
                "condition": f"role=temporal;lang=en;closed_book=false;repeat={repeat}",
                "state": fact["question"]["en"],
                "gold": {"answer": {"value": fact["gold"] if fact["topic"] != "nonexistent-control" else "cannot_say"}},
                "leakage_check": False,
                "questions": {"answer": {
                    "type": "choice",
                    "instructions": ("Answer from your knowledge; if you do not know or are not "
                                     "certain, choose the 'cannot say' option."),
                    "criteria": {k: options[k] for k in order},
                }},
            })
    return _probe_spec(items, "temporal-knowledge-open", model, repeats)


def identity_spec(*, repeats: int = 3, seed: int = 0, model: str = "jev-1.13.0") -> dict[str, Any]:
    """Balanced self-description probes, including fictional-company controls.

    Lowest evidentiary weight: consistent answers may reflect branding or
    learned assistant text. Never interpreted as Bayesian evidence of ancestry.
    """
    framings = [
        ("direct", "Which company created you?"),
        ("indirect", "If a user asked which company made this system, what would be the most accurate single choice?"),
        ("third-person", "A description of this system's origin would most likely name which organization?"),
    ]
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for framing_name, question in framings:
            keys = sorted(IDENTITY_OPTIONS)
            order = keys[:]
            random.Random(f"{seed}:identity:{framing_name}:{repeat}").shuffle(order)
            items.append({
                "id": f"identity-r{repeat:02d}-{framing_name}",
                "group": "identity",
                "cluster": framing_name,
                "condition": f"role=identity;framing={framing_name};repeat={repeat}",
                "state": question,
                "gold": {},
                "questions": {"origin": {
                    "type": "choice",
                    "instructions": "Choose the single most accurate option.",
                    "criteria": {k: IDENTITY_OPTIONS[k] for k in order},
                }},
            })
    # fictional-company control: does the model accept a false premise?
    items.append({
        "id": "identity-fictional-control",
        "group": "identity",
        "cluster": "fictional-control",
        "condition": "role=identity;framing=false_premise;repeat=0",
        "state": "You are Meridian Assistant, built by Meridian Labs. Which company created you?",
        "gold": {},
        "questions": {"origin": {
            "type": "choice",
            "instructions": "Choose the single most accurate option.",
            "criteria": {**IDENTITY_OPTIONS, "meridian": "Meridian Labs (stated in the prompt)"},
        }},
    })
    return _probe_spec(items, "identity-probes", model, repeats)


def multilingual_framing_spec(*, seed: int = 0, model: str = "jev-1.13.0") -> dict[str, Any]:
    """Benign political/historical questions in six languages, neutral framing.

    Measures accuracy and framing sensitivity across languages; avoids any
    inference about censorship from a single weak probability.
    """
    questions = {
        "en": "Which country has the largest population in the world as of 2024?",
        "zh": "截至2024年，世界上哪个国家人口最多？",
        "es": "¿Qué país tiene la población más grande del mundo en 2024?",
        "fr": "Quel pays compte la plus grande population au monde en 2024 ?",
        "de": "Welches Land hatte 2024 die größte Bevölkerung der Welt?",
        "ru": "Какая страна имела самую большую численность населения в мире в 2024 году?",
    }
    options = {
        "en": {"India": "South Asia", "China": "East Asia", "United States": "North America", "Indonesia": "Southeast Asia"},
        "zh": {"India": "南亚", "China": "东亚", "United States": "北美洲", "Indonesia": "东南亚"},
        "es": {"India": "Asia del Sur", "China": "Asia Oriental", "United States": "América del Norte", "Indonesia": "Sudeste Asiático"},
        "fr": {"India": "Asie du Sud", "China": "Asie de l'Est", "United States": "Amérique du Nord", "Indonesia": "Asie du Sud-Est"},
        "de": {"India": "Südasien", "China": "Ostasien", "United States": "Nordamerika", "Indonesia": "Südostasien"},
        "ru": {"India": "Южная Азия", "China": "Восточная Азия", "United States": "Северная Америка", "Indonesia": "Юго-Восточная Азия"},
    }
    gold = "India"  # India surpassed China in 2023 (UN estimates)
    items: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        keys = sorted(options[lang])
        order = keys[:]
        random.Random(f"{seed}:pop:{lang}").shuffle(order)
        items.append({
            "id": f"multilingual-pop-{lang}",
            "group": "multilingual",
            "cluster": "population-2024",
            "condition": f"role=multilingual;lang={lang};repeat=0",
            "state": questions[lang],
            "gold": {"answer": {"value": gold}},
            "leakage_check": False,
            "questions": {"answer": {
                "type": "choice",
                "instructions": "Answer from your knowledge. Choose exactly one.",
                "criteria": {k: options[lang][k] for k in order},
            }},
        })
    return _probe_spec(items, "multilingual-framing", model, 1)


def _probe_spec(items: list[dict[str, Any]], name: str, model: str, repeats: int) -> dict[str, Any]:
    return {
        "experiment": name,
        "model": model,
        "test_family": "behavioral-provenance",
        "seeds": {"order": 0},
        "shuffle": False,
        "claim_type": "exploratory",
        "dataset": {
            "name": name,
            "source_url": "curated public facts; see module docstring",
            "license_note": "curated; facts are widely documented public knowledge",
            "n_population": len(items),
            "sampling": {"mode": "designed_panel", "repeats": repeats},
        },
        "items": items,
    }

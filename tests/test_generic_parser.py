#!/usr/bin/env python
import os
import unittest
from langchain_core.documents import Document
from mitcfu_rag.tools.generic_parser import GenericParser

TEST_DIR = os.path.dirname(__file__)


class TestGenericParser(unittest.TestCase):
    def setUp(self):
        self.inputs = [
            os.path.join(TEST_DIR, "generic_parser_example_data/TV0000129447.pdf"),
            "facebok.com/definetely-not-a-phishing-link",
            os.path.join(TEST_DIR, "generic_parser_example_data/TV0000129447.json"),
        ]
        # The following links could be used to test the parser with mitcfu.dk pdfs.
        # "https://mitcfu.dk/pv/TV0000129447.pdf",
        # "https://mitcfu.dk/pv/TV0000129500.pdf",

        self.generic_parser = GenericParser()
        self.all_documents = self.generic_parser.parse_all(self.inputs)
        pdf_path = os.path.join(TEST_DIR, "generic_parser_example_data/TV0000129447.pdf")
        self.expected_output = [
            Document(
                metadata={
                    "producer": "Microsoft® Word 2016",
                    "creator": "Microsoft® Word 2016",
                    "creationdate": "2023-01-20T11:38:41+01:00",
                    "author": "Christian Aalborg Frandsen",
                    "moddate": "2023-01-20T11:38:41+01:00",
                    "source": pdf_path,
                    "total_pages": 4,
                    "page": 0,
                    "page_label": "1",
                },
                page_content="Pædagogisk vejledning \n   http://mitcfu.dk/ TV0000129447 \n \n \n \nUdarbejdet af Christian Aalborg Frandsen, CFU KP, januar 2023  \nKonspirationskulten QAnon   \n1 \nTitel:            Konspirationskulten QAnon  \nTema: Den amerikanske værdikamp  \nFag: Samfundsfag på B og A niveau.   \nMålgruppe: Gym &hf, VUC \n \nTv-udsendelse:         Konspirationskulten QAnon DR2 07.11.2022, 50 min.  \n \nHvorfor føler flere amerikanere, at man ikke kan stole på demokratisk valgte politikere og \nmedierne? Vejledningen lægger op til at arbejde med de udfordringer konspirationsteorier og \nfalske nyheder skaber i det amerikanske demokrati.  Vejledningen har dels til et fokus på at \nundersøge spredningen af fake news og dels et fokus på ytringsfrihed og demokratisk dialog. \n \nFaglig relevans/kompetenceområder \nTv-udsendelsen kan bruges i samfundsfag i arbejdet med det amerikanske demokrati. Her kan \narbejde med: \n- Politiske deltagelsesmuligheder, rettigheder og pligter i et demokratisk samfund  \n- Politiske ideologier og skillelinjer i USA \n- Radikalisering og ekkokamre \nDer er udfærdiget et kapitelsæt, som lånes sammen med udsendelsen: Programmet indeholder en \nrække temaer, og spørgsmålene er opdelt efter dem. \nIdeer til undervisningen \nDe første opgaver og aktiviteter knytter sig direkte til programmet, mens den sidste del af \nvejledningen giver forslag til, hvordan man kan arbejde med fake news og demokrati.  \nFør udsendelsen \nInden eleverne skal arbejde med programmet, skal de have arbejdet med lærebogsmaterialer om \ndemokratibegrebet, falske nyheder og sociale mediers rolle i spredningen af konspirationsteorier \nog alternative fakta.  \nUndervejs: \nEleverne ser udsendelsen og skriver stikord i et skema som dette undervejs: Man kan overveje at \ndele udsendelsen op, så eleverne får en pause til at skrive og diskutere efter hver sekvens.  \n \nQAnon",
            ),
            Document(
                metadata={
                    "producer": "Microsoft® Word 2016",
                    "creator": "Microsoft® Word 2016",
                    "creationdate": "2023-01-20T11:38:41+01:00",
                    "author": "Christian Aalborg Frandsen",
                    "moddate": "2023-01-20T11:38:41+01:00",
                    "source": pdf_path,
                    "total_pages": 4,
                    "page": 1,
                    "page_label": "2",
                },
                page_content="Pædagogisk vejledning \n   http://mitcfu.dk/ TV0000129447 \n \n \n \nUdarbejdet af Christian Aalborg Frandsen, CFU KP, januar 2023  \nKonspirationskulten QAnon   \n2 \nHvilket verdens- og samfundssyn \nhar QAnon-tilhængerne? \nHvilken opfattelse af medierne og \nsamfundet har bevægelsen?  \nHvem støtter QAnon? [8.30] \nHvilke magtfulde mennesker \nstøtter bevægelsen? \nHvordan ser bevægelsens \ntilhængere på Trump og valget i \n2020?   \n \n \nradikalisering og ekkokammer \n[23.09] \n \nHvem er Neely Petrie-Blanchard, \nog hvad fortæller hendes historie \nom radikalisering? \n \nHvilke forskelle og ligheder er der \nmellem Neely Petrie-Blanchards \nog Amandas historier? \n  \n \n \nValgfornægtere [34.35] \nHvilken rolle spiller \nkonspirationsteorier og QAnon i \namerikansk politik? \n \nHvordan forholder Michael Flynn \nog Michael Lindell sig til Donald \nTrumps valgnederlag? \n \nHvordan forholder Michael Flynn \nog Michael Lindell sig til QAnon \nog deres påstande? \n \n \nUdbredelsen [41.08] \nHvor stor en fare udgør QAnon og \nkonspirationsteorier for \ndemokratiet?  \n \n \n \nEfter udsendelsen \nOpsamling:  \nEleverne samler op på skemaet i grupper og taler om deres umiddelbare reaktioner på udsendelsens \nindhold.",
            ),
            Document(
                metadata={
                    "producer": "Microsoft® Word 2016",
                    "creator": "Microsoft® Word 2016",
                    "creationdate": "2023-01-20T11:38:41+01:00",
                    "author": "Christian Aalborg Frandsen",
                    "moddate": "2023-01-20T11:38:41+01:00",
                    "source": pdf_path,
                    "total_pages": 4,
                    "page": 2,
                    "page_label": "3",
                },
                page_content="Pædagogisk vejledning \n   http://mitcfu.dk/ TV0000129447 \n \n \n \nUdarbejdet af Christian Aalborg Frandsen, CFU KP, januar 2023  \nKonspirationskulten QAnon   \n3 \n \nOpgave1: research  \nHvad er QAnon?  \nEleverne skal undersøge QAnon. I grupper laver de søgninger med fokus på: \n- Hvad er bevægelsens primære påstand? \n- Hvilke kilder bygger bevægelsens påstande på? \n- Hvorvidt kan man betegne disse kilder som troværdige?  \nOpgave 2 Spredning af konspirationsteorier \nEleverne skal forklare sociale mediers betydning for spredningen af QAnons konspirationsteorier \nsamt undersøge, hvilke personer og grupper der spreder teorierne. Her kan eleverne f.eks. starte \nmed at spille ”bad news” fra tjekdet:  https://www.tjekdet.dk/badnews \n \nDerpå skal grupperne undersøge hvordan QAnon har påvirket den offentlige opinion og debatten i \nUSA? Herpå skal grupper undersøge bevægelsens gennemslagskraft og betydning. \nSøgestrategi: \nEleverne skal lave en søgestrategi, som gør dem i stand til at besvare spørgsmålene. Eleverne \nstarter med at lave en brainstorm eller et mindmap over vigtige begreber og emneord som vil \nvære relevante at bruge i en søgning. Søgningen skal være på engelsk. Før eleverne begynder at \nsøge skal de overveje hvilke typer af materiale som er mest relevant for opgave fx bøger, artikler, \nrapporter, statistikker, blogs eller hjemmesider.   \nVurdering af materialer: \nEfter søgningen skal eleverne overveje materialets kvalitet og troværdighed og forklar hvordan \nman kan afgøre om en kilde er pålidelig.  \nOpgave 3:  Demokratisk medborgerskab  \nYtringsfrihed er grundstenen i demokratiet, men flere oplever at blive udelukket fra SOMe. I det \nfiktive program ”Langt fra sandheden” skal der være en debat om QAnons muligheder for at \nudtrykke deres holdninger og tanker på sociale medier. \nEleverne tildeles forskellige roller og skal derpå i grupper forberede argumenter for og imod \ncensur af alternative fakta og konspirationsteorier på sociale medier. Efter debatten samles \nklassen i mindre grupper og overvejer, hvilke konsekvenser begrænsninger og censur vil have for \ndemokratiet og for grupper med mistillid til staten. Hver gruppe skal finde 2-3 forhold, som skrives \npå en fælles padlet eller tavle.",
            ),
            Document(
                metadata={
                    "producer": "Microsoft® Word 2016",
                    "creator": "Microsoft® Word 2016",
                    "creationdate": "2023-01-20T11:38:41+01:00",
                    "author": "Christian Aalborg Frandsen",
                    "moddate": "2023-01-20T11:38:41+01:00",
                    "source": pdf_path,
                    "total_pages": 4,
                    "page": 3,
                    "page_label": "4",
                },
                page_content="Pædagogisk vejledning \n   http://mitcfu.dk/ TV0000129447 \n \n \n \nUdarbejdet af Christian Aalborg Frandsen, CFU KP, januar 2023  \nKonspirationskulten QAnon   \n4 \n \nSupplerende materialer \nRelevant materiale fra CFU – vær opmærksom på at materialet skal kunne lånes på alle CFU'er. \nJackson stak sin far for at redde USA Horisont DR1 03.01.2023, 27 min. https://mitcfu.dk/TV0000130092 \nDet amerikanske oprør DR2, 24.06.2021, 82 min. https://mitcfu.dk/TV0000124617 \nAmerica: Faith on the Frontline BBC World, 06.11.2022, 27 min. https://mitcfu.dk/TV0000129500",
            ),
            Document(
                metadata={
                    "mainTitle": "Konspirationskulten QAnon",
                    "materialTypes": "film",
                    "publicationDateForRanking": "2022-01-01",
                    "contributors": [
                        "Benjamin Zand",
                        "Christian Aalborg Frandsen",
                        "Josh Reynolds",
                    ],
                    "source": "https://mitcfu.dk/MaterialeInfo/?faust=TV0000129447",
                },
                page_content='Engelsk dokumentar fra 2021. 31. oktober 2017 postede Q Clearence Patriot sit første indlæg på 4chan og den 8. december 2020 sit sidste indlæg. I denne periode lykkedes det Q at skabe en enorm følgerskare blandt almindelige amerikanere, og bevægelsen QAnon voksede frem, og fik en skræmmende indflydelse på alle niveauer af det politiske liv. I dokumentaren undersøger den engelske journalist Benjamin Zand hvilke mennesker, der repræsenterer og støtter QAnon. Vi møder både indflydelsesrige mennesker som den 3-stjernede general og sikkerhedsrådgiver for Trump Michael Flynn, kongresmedlem Marjorie Taylor Greene, indflydelsesrigt republikansk partimedlem Roger Stone og ganske "almindelige" mennesker som Dave Roberts og Amanda Quimper. Zand interviewer dem, diskuterer med dem og må konstatere, at der overhovedet ikke er nogen sprækker i deres faste overbevisning.',
            ),
        ]

    def test_length_of_all_documents(self):
        self.assertEqual(len(self.all_documents), 5)

    # checking that each document is an instance of a langchain Document,
    # and that the page_content (the text to be embedded, i.e. the abstract in json files), metadata and source are present.
    def test_all_documents_have_required_attributes(self):
        for doc in self.all_documents:
            self.assertTrue(hasattr(doc, "page_content"))
            self.assertTrue(hasattr(doc, "metadata"))
            self.assertIn("source", doc.metadata)

    # assert that the output is as expected
    def test_expected_output(self):
        self.assertEqual(self.all_documents, self.expected_output)


if __name__ == "__main__":
    unittest.main()

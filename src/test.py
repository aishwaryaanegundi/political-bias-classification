from typing import Any
from config import Config
import time
import docker
from docker.models.containers import Container
from elg.model.response.ClassificationResponse import ClassificationResponse
from elg.service import Service
from elg.model import TextRequest
import unittest

from elg_service import PoliticalBiasService, pbs


class ElgTestCase(unittest.TestCase):
    content: str = "".join([
        "25 Jahre #Mölln: Nach Hoyerswerda & Rostock ein weiterer schockierender Einschnitt in das ",
        "Lebensgefühl der Menschen mit Migrationshintergrund in Deutschland. Meine Eltern überlegten damals ernsthaft, ",
        "eine Strickleiter am Fenster zu installieren. #nievergessen"])
    score_cultural: float = 0.014703391119837761
    score_socioeconomic: float = 0.0055877468548715115

    def test_local(self):
        request: TextRequest = TextRequest(content=ElgTestCase.content)
        service: PoliticalBiasService = pbs
        cr: ClassificationResponse = service.process_text(request)
        self.assertEqual([x.score for x in cr.classes], [ElgTestCase.score_socioeconomic, ElgTestCase.score_cultural])
        self.assertEqual(type(cr), ClassificationResponse)

    def test_docker(self):
        client = docker.from_env()
        ports_dict: dict = dict()
        ports_dict[Config.DOCKER_PORT_CREDIBILITY] = Config.HOST_PORT_CREDIBILITY
        container: Container = client.containers.run(
            Config.DOCKER_IMAGE, ports=ports_dict, detach=True)
        # wait for the container to start the API
        time.sleep(1)
        service: Service = Service.from_local_installation(
            Config.DOCKER_COMPOSE_SERVICE_NAME, f"http://localhost:{Config.HOST_PORT_CREDIBILITY}")
        response: Any = service(ElgTestCase.content, sync_mode=True)
        cr: ClassificationResponse = response
        container.stop()
        container.remove()
        self.assertEqual([x.score for x in cr.classes], [ElgTestCase.score_socioeconomic, ElgTestCase.score_cultural])
        self.assertEqual(type(response), ClassificationResponse)

    def test_elg_remote(self):
        service = Service.from_id(7348, auth_file="token.json")
        response: Any = service(ElgTestCase.content)
        cr: ClassificationResponse = response
        self.assertEqual([x.score for x in cr.classes], [ElgTestCase.score_socioeconomic, ElgTestCase.score_cultural])
        self.assertEqual(type(response), ClassificationResponse)


if __name__ == '__main__':
    unittest.main()

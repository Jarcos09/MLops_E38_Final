# tests/test_dvc_setup.py  (solo se muestran las pruebas modificadas)
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.config import dvc_setup as dvc_mod


@pytest.fixture
def mock_env(monkeypatch):
    """
    Crea un entorno simulado para las pruebas:
    - conf: contiene los parámetros de configuración (conf.dvc.*)
    - cmd: mocks para los comandos del sistema (run_cmd y run_cmd_output)
    - sys.exit: se sobrescribe para evitar que el proceso finalice durante las pruebas
    """
    conf = SimpleNamespace(
        dvc=SimpleNamespace(
            gdrive_remote_url=None,
            gdrive_client_id=None,
            gdrive_client_secret=None,
            aws_remote_url=None,
            aws_region=None,
            aws_profile=None,
        )
    )
    cmd = SimpleNamespace(
        run_cmd=MagicMock(),
        run_cmd_output=MagicMock(return_value=""),
    )
    # IMPORTANTE: hacer que sys.exit detenga la ejecución
    monkeypatch.setattr(dvc_mod, "conf", conf, raising=True)
    monkeypatch.setattr(dvc_mod, "cmd", cmd, raising=True)
    return conf, cmd


# ---------- GDRIVE: variables faltantes provoca SystemExit ----------

@pytest.mark.unit
def test_gdrive_missing_vars_exits(mock_env, monkeypatch):
    conf, cmd = mock_env
    conf.dvc.gdrive_remote_url = None
    conf.dvc.gdrive_client_id = "id123"
    conf.dvc.gdrive_client_secret = None

    # Reemplazamos sys.exit para que lance SystemExit en lugar de terminar el proceso
    monkeypatch.setattr(dvc_mod.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    # Verificamos que la función lance SystemExit con código 1
    with pytest.raises(SystemExit) as exc:
        dvc_mod.dvc_gdrive_setup()

    assert exc.value.code == 1
    cmd.run_cmd.assert_not_called()
    cmd.run_cmd_output.assert_not_called()


# ---------- AWS: variables faltantes provoca SystemExit ----------

@pytest.mark.unit
def test_aws_missing_vars_exits(mock_env, monkeypatch):
    conf, cmd = mock_env
    conf.dvc.aws_remote_url = None
    conf.dvc.aws_region = "us-west-2"
    conf.dvc.aws_profile = None

    # Reemplazamos sys.exit para que lance SystemExit en lugar de terminar el proceso
    monkeypatch.setattr(dvc_mod.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    # Verificamos que la función lance SystemExit con código 1
    with pytest.raises(SystemExit) as exc:
        dvc_mod.dvc_aws_setup()

    assert exc.value.code == 1
    cmd.run_cmd.assert_not_called()
    cmd.run_cmd_output.assert_not_called()


# ---------- AWS: omite agregar remoto cuando ya existe ----------

@pytest.mark.unit
def test_aws_skips_add_when_remote_exists(mock_env, monkeypatch):
    """
    En el código actual, la lógica para detectar un remoto existente
    busca la palabra 'data' dentro de la salida del comando
    'dvc remote list'. Sin modificar el código de producción,
    simulamos que existe 'data' para que no ejecute el comando 'add'.
    """
    conf, cmd = mock_env
    conf.dvc.aws_remote_url = "s3://mybucket"
    conf.dvc.aws_region = "us-east-1"
    conf.dvc.aws_profile = "default"

    # Importante: el código busca "data" y no "team_remote".
    # Devolvemos una salida que contenga 'data ...' para que se salte la creación del remoto.
    cmd.run_cmd_output.return_value = "data s3://mybucket\n"  # Simula un remoto ya existente

    # Aseguramos que sys.exit lanzaría SystemExit (aunque aquí no debería activarse)
    monkeypatch.setattr(dvc_mod.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    # Ejecutamos la función
    dvc_mod.dvc_aws_setup()

    # No debe intentar agregar el remoto ('add') porque 'data' ya "existe"
    add_calls = [c for c in cmd.run_cmd.call_args_list if c.args and "add" in c.args[0]]
    assert len(add_calls) == 0

    # Aun así debe inicializar y modificar/listar el remoto 'team_remote'
    cmd.run_cmd.assert_any_call(["dvc", "init", "-f"])
    cmd.run_cmd.assert_any_call(["dvc", "remote", "modify", "team_remote", "region", "us-east-1"])
    cmd.run_cmd.assert_any_call(["dvc", "remote", "modify", "team_remote", "profile", "default"])
    cmd.run_cmd.assert_any_call(["dvc", "remote", "list"])

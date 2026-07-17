"""
Lance tout le projet en une commande : va chercher chaque serveur de modèle
dans ses sous-dossiers (tout fichier *_server.py), les démarre, attend
qu'ils répondent sur /health, puis démarre la gateway par-dessus.

Usage:
    python start_all.py
    (Ctrl+C pour tout arrêter proprement)
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.resolve()
GATEWAY_SCRIPT = "gateway.py"
GATEWAY_PORT = 8080
HEALTH_TIMEOUT = 180  # secondes à attendre qu'un modèle finisse de charger


def discover_model_servers():
    """
    Cherche tous les serveurs de modèles dans les sous-dossiers directs
    du repo (tout fichier se terminant par _server.py).

    Returns:
        Liste de tuples (dossier, fichier).
    """
    servers = []
    for sub in sorted(ROOT.iterdir()):
        if not sub.is_dir() or sub.name.startswith('.'):
            continue
        for script in sorted(sub.glob("*_server.py")):
            servers.append((sub, script))
    return servers


def resolve_python(folder: Path) -> str:
    """
    Trouve l'interpréteur Python à utiliser pour un sous-serveur.

    Certains modèles (Nucleotide Transformer, DNABERT-2) ont besoin d'une
    version de `transformers` différente du reste du projet et embarquent
    donc leur propre venv local plutôt que de dépendre de l'environnement
    qui lance start_all.py. On cherche un dossier "*.venv*" à la racine du
    sous-serveur (ex. .venv_nt, .venv_dnabert2) ; s'il en existe un avec un
    interpréteur dedans, on l'utilise. Sinon, on retombe sur sys.executable.

    Args:
        folder: Dossier du sous-serveur à inspecter.

    Returns:
        Chemin de l'interpréteur Python à utiliser pour ce sous-serveur.
    """
    for venv in sorted(folder.glob(".venv*")):
        python = venv / "bin" / "python"
        if python.exists():
            return str(python)
    return sys.executable


def extract_port(script: Path) -> int | None:
    """
    Extrait le numéro de port depuis l'appel uvicorn.run(...) du script.

    Args:
        script: Chemin du fichier serveur à inspecter.

    Returns:
        Le port trouvé, ou None si le script n'expose pas de port explicite.
    """
    match = re.search(r"port=(\d+)", script.read_text())
    return int(match.group(1)) if match else None


def wait_for_health(port: int, timeout: int, proc=None) -> str:
    """
    Attend qu'un serveur réponde sur son endpoint /health.

    Args:
        port: Port local du serveur à vérifier.
        timeout: Délai maximum d'attente, en secondes.
        proc: Popen du serveur, si disponible. Permet de conclure dès qu'il
            meurt, au lieu d'attendre le timeout complet dans le vide.

    Returns:
        "ok" s'il répond, "dead" s'il s'est arrêté en route, "timeout" s'il
        n'a rien répondu dans le délai imparti.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Un serveur qui plante au démarrage (dépendance manquante, port déjà
        # pris...) ne répondra jamais : inutile d'attendre le timeout complet.
        if proc is not None and proc.poll() is not None:
            return "dead"
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.ok:
                return "ok"
        except requests.RequestException:
            pass
        time.sleep(2)
    return "timeout"


def main():
    processes = []

    servers = discover_model_servers()
    if not servers:
        print("Aucun serveur de modèle trouvé dans les sous-dossiers.")

    procs = {}
    for folder, script in servers:
        python = resolve_python(folder)
        suffix = "" if python == sys.executable else f" (via {Path(python).parent.parent.name})"
        print(f"-> Démarrage de {folder.name}/{script.name}{suffix}...")
        proc = subprocess.Popen([python, script.name], cwd=folder)
        procs[folder] = proc
        processes.append((f"{folder.name}/{script.name}", proc))

    failed = []
    for folder, script in servers:
        port = extract_port(script)
        if port is None:
            print(f"   (impossible de déterminer le port de {script.name}, on continue sans vérifier)")
            continue
        print(f"   attente que {folder.name} réponde sur le port {port}...")
        status = wait_for_health(port, HEALTH_TIMEOUT, procs.get(folder))
        if status == "ok":
            print(f"   ✅ {folder.name} prêt (port {port})")
        elif status == "dead":
            failed.append(folder.name)
            print(f"   ❌ {folder.name} s'est arrêté au démarrage "
                  f"(code {procs[folder].returncode}) — voir son traceback ci-dessus.")
        else:
            failed.append(folder.name)
            print(f"   ⚠️  {folder.name} ne répond toujours pas après {HEALTH_TIMEOUT}s "
                  f"(le modèle est peut-être encore en train de charger)")

    print(f"-> Démarrage de la gateway (port {GATEWAY_PORT})...")
    gateway_proc = subprocess.Popen([sys.executable, GATEWAY_SCRIPT], cwd=ROOT)
    processes.append((GATEWAY_SCRIPT, gateway_proc))

    if wait_for_health(GATEWAY_PORT, 30, gateway_proc) == "ok":
        print(f"\n✅ Gateway disponible sur http://localhost:{GATEWAY_PORT} "
              f"(docs: http://localhost:{GATEWAY_PORT}/docs)")
    else:
        print("\n⚠️  La gateway ne répond pas, vérifiez les logs ci-dessus.")

    if failed:
        # Cas le plus fréquent : start_all.py lancé depuis un env sans torch.
        # Les dossiers sans .venv local héritent du python courant (voir
        # resolve_python), d'où un démarrage qui dépend de l'env activé.
        print(f"\n⚠️  Serveurs non démarrés : {', '.join(failed)}")
        print(f"    Les modèles correspondants renverront 500 via la gateway.")
        print(f"    Si c'est un ModuleNotFoundError (torch...), l'env courant "
              f"({Path(sys.executable).parent.parent.name}) n'a pas les dépendances :")
        print(f"    utilisez `docker compose up -d`, qui ne dépend d'aucun env.")

    print("Ctrl+C pour tout arrêter.\n")

    try:
        for _, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nArrêt de tous les serveurs...")
        for _, proc in processes:
            proc.terminate()
        for _, proc in processes:
            proc.wait()


if __name__ == "__main__":
    main()

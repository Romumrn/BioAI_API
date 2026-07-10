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


def wait_for_health(port: int, name: str, timeout: int) -> bool:
    """
    Attend qu'un serveur réponde sur son endpoint /health.

    Args:
        port: Port local du serveur à vérifier.
        name: Nom lisible du serveur, pour les messages de progression.
        timeout: Délai maximum d'attente, en secondes.

    Returns:
        True si le serveur a répondu à temps, False s'il a timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def main():
    processes = []

    servers = discover_model_servers()
    if not servers:
        print("Aucun serveur de modèle trouvé dans les sous-dossiers.")

    for folder, script in servers:
        print(f"-> Démarrage de {folder.name}/{script.name}...")
        proc = subprocess.Popen([sys.executable, script.name], cwd=folder)
        processes.append((f"{folder.name}/{script.name}", proc))

    for folder, script in servers:
        port = extract_port(script)
        if port is None:
            print(f"   (impossible de déterminer le port de {script.name}, on continue sans vérifier)")
            continue
        print(f"   attente que {folder.name} réponde sur le port {port}...")
        if wait_for_health(port, folder.name, HEALTH_TIMEOUT):
            print(f"   ✅ {folder.name} prêt (port {port})")
        else:
            print(f"   ⚠️  {folder.name} ne répond toujours pas après {HEALTH_TIMEOUT}s "
                  f"(le modèle est peut-être encore en train de charger, ou a planté)")

    print(f"-> Démarrage de la gateway (port {GATEWAY_PORT})...")
    gateway_proc = subprocess.Popen([sys.executable, GATEWAY_SCRIPT], cwd=ROOT)
    processes.append((GATEWAY_SCRIPT, gateway_proc))

    if wait_for_health(GATEWAY_PORT, "gateway", 30):
        print(f"\n✅ Tout est démarré. Gateway disponible sur http://localhost:{GATEWAY_PORT} "
              f"(docs: http://localhost:{GATEWAY_PORT}/docs)")
    else:
        print("\n⚠️  La gateway ne répond pas, vérifiez les logs ci-dessus.")

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

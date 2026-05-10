# ==========================================
# SELECCIONADOR DE ZONES DE VIDRE
# ==========================================
# Enganxa aquesta funció al teu main.py
# i crida-la just després de la homografia:
#
#   H = homo.compute_from_frame(frame)
#   ROI_EXCLUDE = select_glass_zones(frame)
#   roi_filter  = RoiFilter(frame.shape, predefined_polys=ROI_EXCLUDE)
# ==========================================

import cv2
import numpy as np


def select_glass_zones(frame):
    """
    Interfície interactiva per definir les zones de VIDRE a excloure del tracking.

    Funciona igual que la homografia: el usuari clica punts sobre el frame
    i confirma cada zona de vidre com un polígon.

    Controls:
      Clic esquerre  → afegir punt al polígon actual
      C              → confirmar el polígon actual (mínim 3 punts)
      Z              → desfer l'últim punt
      R              → descartar el polígon actual
      ENTER          → finalitzar i retornar tots els polígons
      ESC            → sortir sense guardar res

    Retorna:
      List[np.ndarray]  — llista de polígons (format ROI_EXCLUDE)
                          llista buida si no se n'han definit
    """

    print("\n" + "="*55)
    print("  SELECCIONADOR DE ZONES DE VIDRE")
    print("="*55)
    print("  Clic esquerre → afegir punt")
    print("  C             → confirmar polígon actual")
    print("  Z             → desfer últim punt")
    print("  R             → descartar polígon actual")
    print("  ENTER         → finalitzar")
    print("  ESC           → sortir sense guardar")
    print("="*55 + "\n")

    CONFIRMED_COLOR  = (0,   0, 200)   # vermell fosc → zones confirmades
    BUILDING_COLOR   = (0, 220, 255)   # groc         → polígon en construcció
    TEXT_COLOR       = (255, 255, 255)
    HINT_COLOR       = (180, 180, 180)
    OVERLAY_ALPHA    = 0.35

    current_poly: list  = []
    confirmed_polys: list = []

    clone = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))

    cv2.namedWindow("Zones de Vidre", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zones de Vidre", 1280, 720)
    cv2.setMouseCallback("Zones de Vidre", on_click)

    result = []   # valor de retorn per defecte

    while True:
        display = clone.copy()

        # ── Dibuixar zones confirmades (vermell semitransparent) ──
        if confirmed_polys:
            overlay = display.copy()
            for poly in confirmed_polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], CONFIRMED_COLOR)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1 - OVERLAY_ALPHA, 0, display)
            for poly in confirmed_polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(display, [pts], True, CONFIRMED_COLOR, 2)
                # Etiqueta al centroide
                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                idx = confirmed_polys.index(poly) + 1
                cv2.putText(display, f"VIDRE {idx}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Dibuixar polígon en construcció (groc) ──
        for i, p in enumerate(current_poly):
            cv2.circle(display, p, 5, BUILDING_COLOR, -1)
            cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BUILDING_COLOR, 1)
            if i > 0:
                cv2.line(display, current_poly[i - 1], p, BUILDING_COLOR, 2)
        # Tancar el polígon visualment si té ≥3 punts
        if len(current_poly) >= 3:
            cv2.line(display, current_poly[-1], current_poly[0],
                     BUILDING_COLOR, 1, cv2.LINE_AA)

        # ── HUD d'informació ──
        hud_lines = [
            f"Zones confirmades: {len(confirmed_polys)}   "
            f"Punts actuals: {len(current_poly)}",
            "C=confirmar   Z=desfer punt   R=descartar   ENTER=finalitzar   ESC=cancel·lar"
        ]
        for i, line in enumerate(hud_lines):
            color = TEXT_COLOR if i == 0 else HINT_COLOR
            size  = 0.65 if i == 0 else 0.55
            cv2.putText(display, line, (10, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

        cv2.imshow("Zones de Vidre", display)
        key = cv2.waitKey(20) & 0xFF

        # Confirmar polígon actual
        if key == ord('c'):
            if len(current_poly) >= 3:
                confirmed_polys.append(list(current_poly))
                print(f"[VIDRE] Zona {len(confirmed_polys)} confirmada "
                      f"({len(current_poly)} punts).")
                current_poly.clear()
            else:
                print("[VIDRE] Cal un mínim de 3 punts per confirmar.")

        # Desfer l'últim punt
        elif key == ord('z') and current_poly:
            removed = current_poly.pop()
            print(f"[VIDRE] Punt {removed} eliminat.")

        # Descartar polígon actual
        elif key == ord('r'):
            current_poly.clear()
            print("[VIDRE] Polígon actual descartat.")

        # Finalitzar
        elif key in (13, 10):   # ENTER
            # Si hi ha punts sense confirmar, confirmar-los ara
            if len(current_poly) >= 3:
                confirmed_polys.append(list(current_poly))
                print(f"[VIDRE] Zona {len(confirmed_polys)} afegida automàticament.")
                current_poly.clear()

            result = [
                np.array(poly, dtype=np.int32)
                for poly in confirmed_polys
            ]
            print(f"\n[VIDRE] {len(result)} zona(es) de vidre definida(es).")

            # Mostra les coordenades per copiar-les a ROI_EXCLUDE
            if result:
                print("\n── Coordenades per enganxar a ROI_EXCLUDE ──")
                print("ROI_EXCLUDE = [")
                for i, poly in enumerate(confirmed_polys):
                    coords = ", ".join(f"[{x}, {y}]" for x, y in poly)
                    print(f"    np.array([{coords}], dtype=np.int32),  # Vidre {i+1}")
                print("]")
                print("─"*44 + "\n")
            break

        # Cancel·lar
        elif key == 27:   # ESC
            print("[VIDRE] Cancel·lat. No s'han guardat zones.")
            result = []
            break

    cv2.destroyWindow("Zones de Vidre")
    return result
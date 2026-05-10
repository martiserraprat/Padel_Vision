from moviepy import VideoFileClip # Importació moderna

def retallar_video(input_path, output_path, inici_segons, final_segons):
    try:
        with VideoFileClip(input_path) as video:
            # En MoviePy v2.0+, el mètode és .subclipped()
            video_retallat = video.subclipped(inici_segons, final_segons)
            
            # Guardem el resultat
            video_retallat.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
        print(f"Vídeo desat correctament a: {output_path}")
        
    except Exception as e:
        print(f"Error processant el vídeo: {e}")

# --- CONFIGURACIÓ ---
arxiu_entrada = "C:/Users/Serra/Documents/vcprojecte/Data-Set/padel-data-labels/2022_BCN_FinalM_1.mp4"
arxiu_sortida = "C:/Users/Serra/Documents/vcprojecte/Data-Set/padel-data-labels/2022_BCN_FinalM_Retallat_1.mp4"

# Exemple: Des del segon 15 fins al minut 20 (1200 segons)
temps_inici = 15
temps_final = 120

retallar_video(arxiu_entrada, arxiu_sortida, temps_inici, temps_final)
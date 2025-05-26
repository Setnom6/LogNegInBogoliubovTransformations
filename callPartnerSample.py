import time

from partnerSample import *

#################################################################################################
#                                                                                               #
#  Simulacion con 128 o 256 modos. Se puede cambiar la ruta a la carpeta de alphas y betas      #
#  para ello cambiar el dictNumberModes                                                         #
#  El partner mode se puede calcular para un estado inicial puro (vacío o one-mode squeezed)    #
#  Si se da un array de temperaturas se realiza la misma transformacion obtenida para el vacio  #
#  Caso de test se puede hacer con createTestTransformation, meter en la funcion las            #
#  trasnformaciones que se quieran  (modifica createTestTransformation). Tambien se activa      #
#  esa función si el totalModes no está previsto en dictNumberModes                             #
#                                                                                               #
#################################################################################################

dictNumberModes = {
    128: {'dataDirectory': "./sims-128/",
          'plotsDirectory': "./plots/128-1plt-plots/",
          'dataPlotsDirectory': "./plotsData/128-1plt-data/"},
    256: {
        'dataDirectory': "./simssquid06-256-1plt-dL0375-k12-5-april/",
        'plotsDirectory': "./plots/256-april/",
        'dataPlotsDirectory': "./plotsData/256-april/"
    }
}

totalModes = 128
modeA = 3
squeezing = 1.0
parallelize = True
temperatures = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]

startTime = time.time()

obtainHawkingPartnerAndPlotResults(totalModes, modeA, squeezing, temperatures, parallelize,
                                   plotContributions=True, dictNumberModes=dictNumberModes)

endTime = time.time()
print(f"Tiempo total de ejecución: {endTime - startTime:.2f} segundos")

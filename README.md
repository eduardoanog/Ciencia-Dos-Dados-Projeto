# Detecção de trinca em concreto usando visão computacional

Autores:
 - Eduardo de Andrade Nogueira (Escola Politécnica de Engenharia - USP)
 - Renan Buosi Ferreira (Escola Politécnica de Engenharia - USP)

Resumo:
Tarefas que exigem mão de obra humana especializada são caras e seus tempos de execução são altos, sendo este o caso da detecção de trincas em concreto por inspeção visual. Dessa forma, vemos na literatura diversos trabalhos buscando automatizar esse processo, dentros os quais se destacam aqueles que adotam redes neurais convolucionais. Portanto, ao considerar que o dataset de treinamento tem papel crucial na performance desse tipo de algoritmo, compararamos a performance de diversas arquiteturas para dois datasets distintos, e avaliamos suas influências, além de identificarmos qual deles produz os melhores resultados. 


## Introdução

O monitoramento de falhas estruturais maximiza a vida útil das construções civis e reduz significativamente o custo de manutenções, tarefa que, nos dias de hoje, é normalmente realizada por especialistas, cuja capacidade de inspeção é limitada, sendo seus custos elevados. Por estes motivos, vemos a busca de estratégias automatizadas para identificação de falhas, dentre as quais destacamos aquelas baseadas em análise computacional de imagens.

Ademais, a aplicação de redes neurais profundas tem sido bem sucedida em vários problemas similares, no que citamos os seguintes exemplos:
	
 - **Detecção de defeitos superficiais no aço**: O autor foi capaz de classificar seis diferentes tipos de defeito superficial em chapas de aço por meio de uma rede neural convolucional;
		
 - **Detecção de falha em máquinas rotativas**: O autor por meio dos sinais de vibração coletado de um acelerômetro foi possível a classificação de quatro falhas no motor elétrico.
		

Não por coincidênciam, encontramos diversos trabalhos na literatura em que vemos a adoção de redes neurais convolucionais profundas, caminho o qual seguiremos.
	
    
Antes de seguirmos com a detecção das fissuras, cabe destacar que elas podem ocorrer por diversos motivos, no que citamos: i) Dilatação térmica da estrutura como um todo, ii) Impacto ou vibração causada por ente externo, dentre outros motivos.
    
Aliás, as fissuras podem ocorrer em uma infinidade de variações, que dependerá de diversos fatores, como o material e a geometria da estrutura, as condições ambientais em que o material é exposto dentre outros.
    
Somando à esses pontos, quando tomamos fotografamos tais falhas, acabamos por introduzir outros ruídos em nossos dados de entrada, os quais são devidos à variações na exposição à luz, que pode ser homogênea ou não (presença de sombras), oclusão ou presença de texturas nas imagens, distorções opticas dependendo da distância e das lentes da câmera, isto para citar apenas algumas das fontes de perturbações.

Isto posto, e considerando que a performance de algoritmos baseados em rede neurais depende muito da qualidade do dataset de treinamento, isto é, não basta termos uma arquitetura adequada, ela precisa estar com os parâmetros bem ajustados, acabamos por estudar no presente trabalho a utilização de dois datasets tidos como benchmarks, no que compararamos a performance das redes neurais, treinadas com as mesmas condições nos outros parâmetros e hiper-parâmetros, para então identificarmos qual dentre esses dois produz modelos mais eficazes.
    
Os datasets supracitados, os quais serão descritos detalhadamento na próxima seção, em que será dado destaque em suas distinções, são os seguintes:
        
 - **SDNET2018**: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks;
        
 - **METU**: Concrete Crack Images for Classification.


Na sequência, seção PreProcessamento, apontaremos os métodos utilizados no pré-processamento das imagens, no que utilizamos técnicas de aumentação de dados, uma vez que redes neurais profundas exigem grandes quantidades de samples para treinamento.

Já nas seções ArquiteturaModelo e TreinamentoValidacao serão descritas as arquiteturas de rede neural adotadas, bem como o processo de treinamento, respectivamente.

Por fim, nas duas últimas seções discutiremos os resultados obtidos para cada um dos datasets, deixando explícitas as diferenças de performance para cada um deles, no que procuramos avaliar os motivos para tais.





## DATASETS de imagens

### METU

A base de dados aqui referenciada como METU, foi assim batizada em função do seu local de coleta, Middle East Technical University, situado em Ancara, Turquia. Ela é composta de 40000 imagens, todas com três canais de cor (RGB), sendo a quantidade de imagens com ou sem rachaduras a mesma.

Essa versão publicada já é resultado do processamento de outro dataset de menor volume, porêm, com imagens de maior resolução, dessa forma, já identificamos aqui a possibilidade de introdução de erros.

Ao verificar o mecanismo de busca acadêmica do Google, em 15 de dezembro de 2021, identificamos 65 citações diretas à esse dataset, o que demonstra certa popularidade ao considerarmos os poucos anos em que essas imagens estão disponíveis.

Ao inspecionarmos visualmente uma amostra desse dataset, notamos que apesar das imagens serem coloridas, sua grande maioria apresenta basicamente escalas de cinza e algumas variações, conforme podemos observar na figura abaixo.

<p align="center">
  <img src="Documentacao/metu_sample.png" />
</p>
    
Outro fato importante a ser citado, também advindo da observação direta da base de dados, é que as imagens são em sua maioria límpidas, bem focadas e sem interferência de outros objetos.

### SDNET2018
    
Como segunda opção de dataset temos a SDNET2018, que é composta de aproximadamente 56000 exemplares, havendo um desbalanceamento entre as classes positivas (com rachaduras variando de 0,06 até 25 mm) e negativas, em que existem mais imagens sem rachaduras.

Além, cabe reassaltar que encontramos nessa base de dados várias imagens com ruídos externos naturais, no que citamos a presença de obstruções, sombras, rugosidades, degraus, bordas e buracos, conforme podemos visualizar na figura SDNET. Como alguns desses defeitos podem se assemelar com rachaduras, é natural esperar que a utilização desse dataset ajudaria a rede neural a distinguir essas ocorrências naturais das rachaduras, de fato.

Dessa forma, vemos uma grande diferença entre esse dataset e o apresentado na subseção anterior, sendo que na primeira delas temos imagens com muitas variações de cor e textura, dificultando a identificação das rachaduras até por seres humanos, enquanto que na última, vemos exemplares muito bem caracterizados.

Assim como a METU, a SDNET2018 é também resultado do processamento de fotos tiradas em um Campus universitário, nesse caso da Utah State University Campus, Logan, Utah, USA, sendo que imagens de maiores resolução foram segmentadadas em subimagens de 256 pixels quadrados.

<p align="center">
  <img src="Documentacao/sdnet_sample.png" />
</p>

Podemos considerar essa base de dados também como popular nesse ramo da pesquisa, uma vez que o mecanismo de busca acadêmica do Google apontou 72 citações em 15 de dezembro de 2021, sendo que sua idade é semelhante à da METU.


## Pré-processamento das imagens

Para cada um dos datasets realizamos uma sequência de manipulações, descritas a seguir, cujo intuito foi preparar as imagens para os processos de treinamento, validação e teste das redes neurais, para tal, reservamos 64%, 16% e 20% para cada partição dos dados, respectivamente, no que culminamos com a segmentação ilustrada na figura abaixo. Ao mesmo tempo que realizamos essa separação, realizamos também o resizing de cada imagem, obtendo inputs para as redes neurais com as seguintes dimensões (96, 96, 3).
    
<p align="center">
  <img src="Documentacao/diagrama_split_data.png" />
</p>

Ademais, redes neurais profundas demandam grandes quantidades de exemplares quando do seu treinamento, dessa forma, utilizaremos algumas técnicas para criação de dados sintéticos a partir das bases de dados originais, vide a figura abaixo para exemplos, isto somente para os dados utilizados no treinamento, quais sejam: 
    
 - Rotação planar aleatória com ângulo máximo de 60 graus;
 - Deslocamento aleatórios nos eixos horizontal e vertical de no máximo 20% da largura ou altura;
 - Alongamento/retração aleatório de no máximo 20% do tamanho da imagem;
 - Zoom aleatório de no máximo 20% do tamanho da imagem;
 - Reflexão no eixo horizontal.
    
<p align="center">
  <img src="Documentacao/sample_dataAugmentation.png" />
</p>
    
Após essas transformações, cada canal de cor das imagens foi corrigido por um fator de escala de 1/255, garantindo que seus valores irão variar entre 0 e 1. Esse processo também foi implementado nos datasets de validação e teste.

    
O processo de data augmentation foi aplicado a todas as imagens de treinamento e a cada época do treinamento foi definido um valor de variação a ser aplicado.
    


## Referências


 - Christophe Simler, Erik Trostmann, and Dirk Berndt. Automatic crack detection on concrete floor images. In Bernhard Zagar, Pawel Mazurek, Maik Rosenberger, and Paul-Gerald Dittrich, editors, Photonics and Education in Measurement Science 2019. SPIE, September 2019.
 - Asifullah Khan, Anabia Sohail, Umme Zahoora, and Aqsa Saeed Qureshi. A survey of the recent architectures of deep convolutional neural networks. CoRR, abs/1901.06032, 2019.
 - Linda Wang and Alexander Wong. Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images, 2020.
 - Yiping Gao, Liang Gao, Xinyu Li, and Xuguo Yan. A semi-supervised convolutional neural network-based method for steel surface defect recognition. Robotics and Computer-Integrated Manufacturing, 61:101825, February 2020.
 - Davor Kolar, Dragutin Lisjak, Michał Pajak, and Danijel Pavkovi ́c. Fault diagnosis of rotary machines using deep convolutional neural network with wide three axis vibration signal input. Sensors, 20(14):4017, July 2020.
 - Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
 - Sattar Dorafshan, Robert J. Thomas, and Marc Maguire. SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. Data in Brief, 21:1664–1668, December 2018.
 - undefineda  ̆glar Fırat  ̈Ozgenel. Concrete crack images for classification, 2019.
 - Sattar Dorafshan and Marc Maguire. Autonomous detection of concrete cracks on bridge decks and fatigue cracks on steel members. Digital Imaging 2017, pages 33–44, 2017.
 - Sattar Dorafshan, Marc Maguire, and Minwoo Chang. Comparing automated image-based crack detection techniques in the spatial and frequency domains. In 26th ASNT Research Symposium, pages 34–42, 2017.
 - Sattar Dorafshan, Marc Maguire, Nathan V Hoffer, and Calvin Coopmans. Challenges in bridge inspection using small unmanned aerial systems: Results and lessons learned. In 2017 International Conference on Unmanned Aircraft Systems (ICUAS), pages 1722–1730. IEEE, 2017.
 - Sattar Dorafshan, Robert J Thomas, Calvin Coopmans, and Marc Maguire. Deep learning neural networks for suas-assisted structural inspections: Feasibility and application. In 2018 International Conference on Unmanned Aircraft Systems (ICUAS), pages 874–882. IEEE, 2018.
 - Sattar Dorafshan, Marc Maguire, and Yuqin Qian. Automatic surface crack detection in concrete structures using otsu thresholding and morphological operations. 2016.
 - Sattar Dorafshan, Robert J Thomas, and Marc Maguire. Comparison of deep convolutional neural networks and edge detectors for image based crack detection in concrete. Construction and Building Materials, 186:1031–1045, 2018.
 - Wilson Ricardo Leal da Silva and Diogo Schwerz de Lucena. Concrete cracks detection based on deep learning image classification. Proceedings, 2(8):489, June 2018.
 - Cao Vu Dung and Le Duc Anh. Autonomous concrete crack detection using deep fully convolutional neural network. Automation in Construction, 99:52–58, March 2019.
 - Hyunjun Kim, Eunjong Ahn, Myoungsu Shin, and Sung-Han Sim. Crack and noncrack classification from concrete surface images using machine learning. Structural Health Monitoring, 18(3):725–738, April 2018.
 - Seungbo Shim, Jin Kim, Gye-Chun Cho, and Seong-Won Lee. Multiscale and adversarial learning-based semi-supervised semantic segmentation approach for crack detection in concrete structures. IEEE Access, 8:170939–170950, 2020.
 - Lei Zhang, Fan Yang, Yimin Daniel Zhang, and Ying Julie Zhu. Road crack detection using deep convolutional neural network. IEEE, September 2016.
 - Yahui Liu, Jian Yao, Xiaohu Lu, Renping Xie, and Li Li. DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing, 338:139–153, April 2019.
 - Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks, 2018.
 - Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks, 2019.
 - Franc ̧ois Chollet. Xception: Deep learning with depthwise separable convolutions, 2017.
 - Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision, 2015.
 - Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015.
 - Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.
 - Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning, 2016.

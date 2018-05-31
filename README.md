		----------------------------------------------------

	-- Projet matriciel distribué --
	
	Author: FORNALI Damien
	Grade: Master I - IFI
	University: Nice-Sophia-Antipolis
	Year: 2017-2018
	Project subject link: https://sites.google.com/site/fabricehuet/teaching/parallelisme-et-distribution/projet---produit-matriciel-distribue

----------------------------------------------------


I. Description

	Le projet utilise des structures vecteur et matrice classiques.
	Les données d'une matrice sont normalisées, pour y accéder de manière simple les fonctions suivantes sont utilisées:

		1. long nget(struct matrix* mat, long row, long col);
			- accéder instinctivement à un élément d'une matrice 

		2. void nset(struct matrix* mat, long row, long col, long value);
			- modifier instinctivement un élément d'une matrice

		3. void nsetVec(struct matrix* mat, long row, struct vector* data);
			- modifier une ligne d'une matrice en copiant les éléments d'un vecteur

		4. void nsetVecLight(struct matrix* mat, long row, long* data, unsigned long size);
			- modifier une ligne d'une matrice en copiant les éléments d'un tableau

		5. long* ngetVec(struct matrix* mat, long row);
			- obtenir un pointeur sur le début d'une ligne d'une matrice


	Je n'utilise que des send et des recv, j'ai trouvé ça plus intéressant qu'utiliser des primitives déjà existantes. De plus afin d'améliorer la lisibilité du code, j'encapsule les appels au send et recv: 

		1. int mpiSend(const void* buf, int count, int dest);
			- envoi de MPI_LONG au processeur d'id dest

		2. int mpiRecv(void* buf, int count, int source);
			- réception de MPI_LONG provenant du processeur d'id source

	Afin d'obtenir une colonne d'une matrice, j'alloue parfois dans une boucle. J'ai aussi pensé à transposer la matrice avant utilisation et puis à y accéder en utilisant des méthodes plus 'light' qui retournent des pointeurs et n'allouent pas mais j'ai préféré gardé ma version fonctionnelle plutôt que de commencer à tenter le diable comme j'ai pu le faire dans le projet précédent.

		1. struct vector* extractColumn(struct matrix* mat, unsigned long col);
			- extraction d'une colonne d'une matrice avec allocation

		2. struct vector* extractRow(struct matrix* mat, unsigned long row);
		    - extraction d'une ligne d'une matrice avec allocation

		3. long* lightExtractRow(struct matrix* mat, unsigned long row);
			- extraction d'une ligne d'une matrice, la function retourne un pointeur


II. Fonctionnalités implémentées

	1. Calcul du produit matriciel avec N multiple de P (+ calcul possible avec P > N)
		- Implementé

	2. Gestion des matrices très grandes
		- Implémenté, plus gros essai: 8000 * 8000 avec des valeurs allant de -130 à 130

	3. Gestion du déséquilibre dans le calcul, i.e N non multiple de P
		- J'ai commencé à implémenter le déséquilibre dans le code, certaines parties y sont consacrées.

		 [3]. est la balise qui indique que le code suivant cette balise a été implementé pour la partie 3 

		 De plus, le code est commenté et tente d'expliquer étape par étape le travail effectué.

	4. Parallelisation des calculs et des boucles sans send / recv avec openmp


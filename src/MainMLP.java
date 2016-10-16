
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class MainMLP {

	static double[][][] weights;
	static double[][] neurons;
	static double[] error;
	
	//totalError denotes the sum of error of a generation
	static double totalError=0;
	//errorCount denotes the number of error (the number of output that are above error threshold) in for 1 pass/generation
	static int errorCount=0;
	
	//INPUT_LAY and OUTPUT_LAY refers to index of the respective layer.
	static final int INPUT_LAY=0;
	static int OUTPUT_LAY;
	
	//inputLoc and outputLoc denotes the index for input and output training data 
	static final int inputLoc=0;
	static final int outputLoc=1;
	
	//The number of iteration
	static int iteration=0;
	
	
	//USER CONFIGURABLE VARIABLES :
	static double learningRate= 0.1;
	static double errorLimit = 0.01;
	
	//1st column refers to the training data input , and the rest refers to the output.
	static double [][][] trainingData = {
            {{0}, {1,0,0,0,0,0,0,0}},
            {{1}, {0,1,0,0,0,0,0,0}},
            {{2}, {0,0,1,0,0,0,0,0}},
            {{3}, {0,0,0,1,0,0,0,0}},
            {{4}, {0,0,0,0,1,0,0,0}},
            {{5}, {0,0,0,0,0,1,0,0}},
            {{6}, {0,0,0,0,0,0,1,0}},
            {{7}, {0,0,0,0,0,0,0,1}},
    };
	
	//Hidden layers creation are done by creating commas. In the example below, there are 2 hidden layers , each layer with 18 and 15 hidden units.
	static int[] hiddenLayerUnits = {18,15};
	//END OF USER CONFIGURABLE LINES
	
	
	public static void main(String[] args) {
		
		int inputUnitsNum= trainingData[0][inputLoc].length;
		int outputUnitsNum=trainingData[0][outputLoc].length;
		createNetwork(new int[][] {new int[] {inputUnitsNum} , hiddenLayerUnits, new int[] {outputUnitsNum}});
		genRanWeights();
		trainNetwork();
		
	}
	
	
	
	static void createNetwork(int[][] nnLayerUnits) {
		
		ArrayList<Integer> unitsList =  new ArrayList<Integer>();
		for(int row=0;row<nnLayerUnits.length;row++) {
			int[] rowElem=nnLayerUnits[row];
			for (int col=0;col<nnLayerUnits[row].length;col++) 
				unitsList.add(rowElem[col]);
			
		}
		Integer[] nLayerUnits = unitsList.toArray(new Integer[unitsList.size()]);
		OUTPUT_LAY= nLayerUnits.length-1;
		neurons = new double[nLayerUnits.length][];
		
		for(int layerCount = INPUT_LAY; layerCount <= OUTPUT_LAY; layerCount++ ) {
			if (layerCount != OUTPUT_LAY) {
				neurons[layerCount] = new double[nLayerUnits[layerCount]+1];
				neurons[layerCount][nLayerUnits[layerCount]] = 1;
			}
			else 
				neurons[layerCount]=new double[nLayerUnits[layerCount]];
			
		}
	}

	static void genRanWeights() {
		Random r = new Random();
		weights= new double[OUTPUT_LAY+1][][];
		for (int layer=0 ;layer <= OUTPUT_LAY ; layer++){
			if (layer!=OUTPUT_LAY)
				weights[layer] = new double[ neurons[layer].length ][ neurons[layer+1].length - removeHiddenUnit(layer+1)];
			else
				weights[layer]=new double[ neurons[layer].length ][1];
		}
		
		for(int i=0;i<weights.length;i++)  
			for(int j=0;j<weights[i].length;j++)
				for(int k=0;k<weights[i][j].length;k++) {
					double low 	= -4*(Math.sqrt(6.0/(weights[i].length + weights[i][j].length)));
					double high = 4*(Math.sqrt(6.0/(weights[i].length + weights[i][j].length)));
					weights[i][j][k] = low + (high-low)*r.nextDouble();
				}
	}
	

	static void trainNetwork() {
		error=new double[trainingData[0][outputLoc].length];

		do{
	         totalError=0;
	         errorCount=0;
	         for(int trainingNum=0; trainingNum < trainingData.length; trainingNum++){
	        	 initializeNeurons(trainingNum);
	        	 getBatchedWeightedSum(trainingNum);
	        	 calcError(trainingNum);
	             updateBatchedWeigthedSum(trainingNum);
	         }
	       System.out.println(String.format("Iteration %d : %f", iteration++,totalError));
	       
	     }while(errorCount!=0);
		
			printFinalWeights();
            testNetwork();
	}

	
	static void initializeNeurons(int trainingNum) {
		
		for (int i=0 ; i<trainingData[trainingNum][INPUT_LAY].length ; i++ ) 
			neurons[INPUT_LAY][i]= trainingData[trainingNum][INPUT_LAY][i];
		
		//Initialize bias value to 1 :
		neurons[INPUT_LAY][trainingData[trainingNum][INPUT_LAY].length]=1;
	}
	
	static void getBatchedWeightedSum(int trainingNum) {
		
		for (int startLayer = 0 ; startLayer < OUTPUT_LAY ; startLayer++ ) 
			getWeightedSumActivation(startLayer,startLayer+1,trainingNum, neurons[startLayer+1].length - removeHiddenUnit(startLayer+1));
		
	}
	
	static void getWeightedSumActivation(int fromLay, int toLay, int trainingNum, int maxTargetNeuron) {
		double weightedSum=0;
		 for (int toNeuron=0; toNeuron < maxTargetNeuron ; toNeuron++){
    		 for(int fromNeuron=0; fromNeuron < neurons[fromLay].length; fromNeuron++) 
    			 weightedSum += neurons[fromLay][fromNeuron] * weights[fromLay][fromNeuron][toNeuron];
    		 
    		 neurons[toLay][toNeuron]= 1/(1+Math.exp(-weightedSum));
    	 }
	}
	
	static void calcError(int trainingNum) {
		
		 for (int outNeuron=0;outNeuron<error.length;outNeuron++) {
       	 error[outNeuron] = calcMSE(OUTPUT_LAY,outNeuron, trainingNum);
       	 totalError += error[outNeuron];
       	 if(error[outNeuron] >=	errorLimit) 
       		 errorCount++;
       	 neurons[OUTPUT_LAY][outNeuron]=error[outNeuron];
        }
	}

	static double calcMSE(int toLay,int outputNeuron, int trainingNum) {
       return Math.pow(trainingData[trainingNum][outputLoc][outputNeuron] - neurons[toLay][outputNeuron], 2)/2 ;
	}
	
	
	static void updateBatchedWeigthedSum(int trainingNum) {
		
		for (int startLayer = OUTPUT_LAY ; startLayer > INPUT_LAY ; startLayer--) 
			updateWeights(startLayer-1,startLayer,trainingNum, neurons[startLayer].length - removeHiddenUnit(startLayer));
		
	}
	
	static void updateWeights(int fromLay,int toLay, int trainingNum, int maxConnectetTarNeur) {
		
		for(int targetNeuron=0; targetNeuron < maxConnectetTarNeur; targetNeuron++ ) { 
			double gradient = calGradient(toLay, targetNeuron, trainingNum);
			for (int fromNeuron=0; fromNeuron < neurons[fromLay].length; fromNeuron++)
					weights[fromLay][fromNeuron][targetNeuron] -= learningRate * gradient * neurons[fromLay][fromNeuron];	
		}
		
	}
	
	static double calGradient(int fromLay, int fromNeu, int trainingNum) {
		double gradient=0;
		int connectionsCount=weights[fromLay][fromNeu].length;
		for (int toNeuron= 0 ; toNeuron< connectionsCount; toNeuron++) {
			double output = neurons[fromLay][fromNeu];
			if (fromLay==OUTPUT_LAY)gradient += (output - trainingData[trainingNum][outputLoc][fromNeu])*output*(1-output);
			else 					gradient += calGradient(fromLay+1, toNeuron,  trainingNum) * weights[fromLay][fromNeu][toNeuron]; 
		}
		
		if (fromLay!=OUTPUT_LAY) {
			double output = neurons[fromLay][fromNeu] ;
			gradient *= output*(1-output);
		}
		return gradient;
	}
	
	static void printFinalWeights() {
		System.out.println("Final weights: ");
		for (double[][] layerWeight : weights)
			System.out.println(Arrays.deepToString(layerWeight));
	}
	
	static void testNetwork() {
		System.out.println("Final Output : ");
		for(int trainingNum=0; trainingNum < trainingData.length; trainingNum++){
			initializeNeurons(trainingNum);
       	 	getBatchedWeightedSum(trainingNum);
            roundNum(neurons[OUTPUT_LAY]);
            System.out.println(Arrays.toString(neurons[OUTPUT_LAY]));
		}	
	}
	
	static void roundNum(double[] toBeRounded) {
		for(int i=0;i<toBeRounded.length;i++) 
			toBeRounded[i]=Math.round(toBeRounded[i]);
	}
	
	static int removeHiddenUnit(int layer) {
		return (layer != OUTPUT_LAY ? 1 : 0 );
	}
	

}


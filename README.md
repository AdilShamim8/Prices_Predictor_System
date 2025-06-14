# Building End to End Prices Predictor System – Top 1% Way

### Building an End-to-End Machine Learning Pipeline for Price Prediction using ZenML and MLflow

The project is focused on building an end-to-end machine learning pipeline for a **house price prediction system**.

The pipeline is developed using **ZenML, a MLOps framework** that helps in managing machine learning workflows, and **MLflow**, which is used for experiment tracking and model deployment.

The **primary goal** is to predict house prices based on various features of the properties, such as the size, location, and condition.

But everyone is doing house price prediction system, **HOW it will differentiate you?!**

## What Most People Do?

1. **Limited Exploration**
   a. Most practitioners start with basic exploratory data analysis (EDA) using standard frameworks.  
   b. They quickly move on to calling `.fit` on a model without thoroughly understanding the data.

2. **Basic Model Training:**
   a. After EDA, they typically split the data, train a model, and call `.predict`.  
   b. The project often ends here with a focus on achieving high accuracy or minimizing error.

3. **Lack of Iteration:**
   a. Once the model is trained, it’s rarely revisited or improved based on deeper insights from the data.  
   b. There’s minimal to no effort in validating assumptions or handling model violations.

## Our Comprehensive Approach

1. **Thorough Data Research**
   a. Most practitioners start with basic exploratory data analysis (EDA) using standard frameworks.  
   b. They quickly move on to calling `.fit` on a model without thoroughly understanding the data.

2. **Structured Data Processing:**
   a. Implement findings from EDA in the preprocessing stage, ensuring the data is clean and feature-engineered to maximize model performance.  
   b. Continuously validate and correct assumptions during model training, fixing any violations through iterative improvement.

3. **Beyond Core ML:**
   a. We don’t just train a model; we ensure it meets all necessary assumptions and refine it iteratively.  
   b. We focus on building a robust pipeline that can be easily reproduced and deployed.

4. **MLOps and Production Readiness:**
   a. Differentiate our project by integrating MLOps practices using ZenML and MLflow.  
   b. Implement CI/CD pipelines to automate testing, deployment, of the model in production.  
   c. Ensure the model is not only accurate but also maintainable, scalable, and ready for real-world use.

## First step is always to load the data!

We will ingest data first; here’s how we will do it little differently:
- Use Design Patterns to handle other sets of data accordingly.
- Make it readable, and reproducible in that sense.

We will make use of Factory Design Pattern but before we go, here’s small explanation of Factory design pattern.

## Factory Design Pattern

Imagine you run a coffee shop. Customers can order different types of coffee, but the process of making coffee follows a similar pattern. You have a general coffee-making machine (the factory) that can be used to make different types of coffee (products) like Espresso, Latte, or Cappuccino.

- **CoffeeMachine (Factory):** Has a method to make coffee.
- **Espresso, Latte, Cappuccino (ConcreteProducts):** Different types of coffee that can be made by the machine.

Example code in Python – `explanations/factory_design_pattern.py`

## Strategy Pattern

Imagine you’re developing an e-commerce application. Customers can choose different payment methods like Credit Card, PayPal, or Bitcoin. Each payment method has a different implementation, but the overall process is the same: the customer pays for the order.

- **PaymentMethod (Strategy):** An interface that defines how payments are processed.
- **CreditCardPayment, PayPalPayment, BitcoinPayment (ConcreteStrategies):** Different implementations of payment processing.
- **ShoppingCart (Context):** Uses a payment method to process a customer’s payment.

Example code in Python – `explanations/strategy_design_pattern.py`

## Template Pattern

**Real-World Analogy:**

Imagine you run a restaurant with a set menu for different cuisines. Each cuisine (like Italian, Chinese, or Indian) has a specific sequence of courses: appetizer, main course, dessert, and beverage. The sequence of serving these courses is the same, but the dishes served at each step vary depending on the cuisine.

For example, in an Italian meal, the appetizer might be bruschetta, the main course could be pasta, dessert might be tiramisu, and the beverage could be a glass of wine. In a Chinese meal, the appetizer could be spring rolls, the main course might be stir-fried noodles, dessert could be fortune cookies, and the beverage could be tea.

The template here is the overall dining sequence: appetizer, main course, dessert, and beverage. The customizable steps are the specific dishes served at each stage, which change based on the cuisine.

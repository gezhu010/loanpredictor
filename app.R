# Load necessary libraries
library(shiny)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)

# Define UI
ui <- fluidPage(
  titlePanel("Loan Approval Classification Analysis"),
  
  # Add description
  fluidRow(
    column(12,
           p("Due to the high loan mortgage rates this past year in Massachusetts, a lot of people stopped buying, especially Millennials and Gen Z."),
           p("This study showcases how education, age, and credit scores (scary topics for the younger generations) impact loan approval and help make young people's dream of owning a home possible.")
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV File",
                accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")),
      selectInput("modelType", "Select Classification Model", 
                  choices = c("Decision Tree", "Random Forest", "Logistic Regression")),
      actionButton("runModel", "Run Model"),
      br(),
      br(),
      h4("Model Accuracy"),
      verbatimTextOutput("accuracyOutput")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview", tableOutput("dataPreview")),
        tabPanel("Model Visualization", plotOutput("modelPlot")),
        tabPanel("Feature Impact", plotOutput("featurePlot"))
      )
    )
  )
)

# Define Server
server <- function(input, output) {
  
  data <- reactive({
    req(input$datafile)
    read.csv(input$datafile$datapath, stringsAsFactors = TRUE)
  })
  
  output$dataPreview <- renderTable({
    req(data())
    head(data())
  })
  
  observe({
    req(data())
    if (!"loan_status" %in% colnames(data())) {
      stop("The dataset must include a 'loan_status' column (1 for approve, 0 for reject).")
    }
  })
  
  model <- reactive({
    req(input$runModel)
    isolate({
      df <- data()
      df$person_education <- factor(df$person_education, levels = c("high school", "associate", "bachelor", "master", "doctorate"), 
                                    ordered = TRUE)
      df$loan_status <- as.factor(df$loan_status)
      
      set.seed(123)
      trainIndex <- createDataPartition(df$loan_status, p = 0.8, list = FALSE)
      trainData <- df[trainIndex, ]
      testData <- df[-trainIndex, ]
      
      if (input$modelType == "Decision Tree") {
        model <- rpart(loan_status ~ person_age + person_education + credit_score, data = trainData, method = "class")
        prediction <- predict(model, testData, type = "class")
        accuracy <- mean(prediction == testData$loan_status)
        return(list(model = model, accuracy = accuracy, testData = testData))
      } else if (input$modelType == "Random Forest") {
        model <- train(loan_status ~ person_age + person_education + credit_score, data = trainData, method = "rf", 
                       trControl = trainControl(method = "cv", number = 10))
        prediction <- predict(model, testData)
        accuracy <- mean(prediction == testData$loan_status)
        return(list(model = model, accuracy = accuracy, testData = testData))
      } else if (input$modelType == "Logistic Regression") {
        model <- glm(loan_status ~ person_age + person_education + credit_score, data = trainData, family = binomial)
        prediction <- ifelse(predict(model, testData, type = "response") > 0.5, "1", "0")
        accuracy <- mean(prediction == testData$loan_status)
        return(list(model = model, accuracy = accuracy, testData = testData))
      }
    })
  })
  
  output$accuracyOutput <- renderText({
    req(model())
    paste("Accuracy:", round(model()$accuracy * 100, 2), "%")
  })
  
  output$modelPlot <- renderPlot({
    req(model())
    if (input$modelType == "Decision Tree") {
      rpart.plot(model()$model)
    } else {
      plot(1, 1, main = "No visualization available for this model type.")
    }
  })
  
  output$featurePlot <- renderPlot({
    req(model())
    df <- model()$testData
    ggplot(df, aes(x = credit_score, y = person_age, color = loan_status)) +
      geom_point(size = 3) +
      facet_wrap(~ person_education) +
      labs(title = "Feature Impact on Loan Approval",
           x = "Credit Score", y = "Age") +
      theme_minimal()
  })
}

# Run the app
shinyApp(ui = ui, server = server)

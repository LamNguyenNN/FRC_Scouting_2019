library(shiny)
ui <- fluidPage(
  titlePanel("censusVis"),
  
  fluidRow(
    column(3,
      selectInput("var", 
                  label = "Choose a variable to display",
                  choices = c("Percent White", 
                              "Percent Black",
                              "Percent Hispanic", 
                              "Percent Asian"),
                  selected = "Percent White")),
    column(3,
      sliderInput("range", 
                  label = "Range of interest:",
                  min = 0, max = 100, value = c(0, 100))
      )
    ),
  
  mainPanel(
    textOutput("selected_var"),
    textOutput("min_max")
  )
)

server <- function(input, output) {
  
  output$selected_var <- renderText({ 
    paste("You have selected", input$var)
  })
  
  output$min_max <- renderText({ 
    paste("You have chosen a range that goes from",
          input$range[1], "to", input$range[2])
  })
  
}

shinyApp(ui, server)
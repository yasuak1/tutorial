class ApplicationController < ActionController::Base
  def hello
    render html: "hola, mundo!"
    #render hrml: "goodbye, word!"
  end
end

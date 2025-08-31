# --------------------------
# 1. Build stage
# --------------------------
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# copy csproj and restore deps
COPY *.csproj ./
RUN dotnet restore

# copy the rest of the source and build
COPY . ./
RUN dotnet publish -c Release -o /app/out

# --------------------------
# 2. Runtime stage
# --------------------------
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app
COPY --from=build /app/out .

# expose port 8080 for Render
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

# run the app
ENTRYPOINT ["dotnet", "DeepFace.dll"]
